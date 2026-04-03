#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

TMP_MPL = Path('/tmp/pslt_mplconfig')
TMP_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(TMP_MPL))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'

OUT_SUMMARY = OUTDIR / 'runtime_direct_detlin_local_fix_summary.csv'
OUT_PNG = OUTDIR / 'runtime_direct_detlin_local_fix_summary.png'
OUT_META = OUTDIR / 'runtime_direct_detlin_local_fix_summary_run_meta.json'

D4P0 = pd.read_csv(OUTDIR / 'runtime_direct_detlin_d4p0_acceptance_only_audit_summary_single_1p309.csv').iloc[0]
D4P8 = pd.read_csv(OUTDIR / 'runtime_direct_detlin_d4p8_single_beta_full_eta_summary_beta0p75.csv').iloc[0]
D64 = pd.read_csv(OUTDIR / 'runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare_summary.csv')
D72 = pd.read_csv(OUTDIR / 'runtime_direct_detlin_tail_single_beta_full_eta_D7p2_summary_beta0p35.csv').iloc[0]
D80 = pd.read_csv(OUTDIR / 'runtime_direct_detlin_tail_single_beta_full_eta_D8p0_summary_beta1p2.csv').iloc[0]
COMBINED = pd.read_csv(OUTDIR / 'runtime_direct_detlin_combined_focus_compare_summary.csv')

rows = [
    {
        'D': 4.0,
        'mechanism': 'local width gate',
        'parameter': 'beta=1.309',
        'step_baseline_p95_abs_delta_mu_mumu': float(106.57226942269449),
        'step_candidate_p95_abs_delta_mu_mumu': float(D4P0['p95_abs_delta_mu_mumu']),
        'step_baseline_acceptance_mismatch': float(0.6097560975609756),
        'step_candidate_acceptance_mismatch': float(D4P0['acceptance_mismatch']),
    },
    {
        'D': 4.8,
        'mechanism': 'local width gate',
        'parameter': 'beta=0.75',
        'step_baseline_p95_abs_delta_mu_mumu': float(D4P8['baseline_p95_abs_delta_mu_mumu']),
        'step_candidate_p95_abs_delta_mu_mumu': float(D4P8['candidate_p95_abs_delta_mu_mumu']),
        'step_baseline_acceptance_mismatch': float(D4P8['baseline_acceptance_mismatch']),
        'step_candidate_acceptance_mismatch': float(D4P8['candidate_acceptance_mismatch']),
    },
    {
        'D': 6.4,
        'mechanism': 'band-pass g_norm',
        'parameter': 'beta=0.25, center=0.06, hw=0.04',
        'step_baseline_p95_abs_delta_mu_mumu': float(D64.loc[np.isclose(D64['D'], 6.4), 'width_only_p95_abs_delta_mu_mumu'].iloc[0]),
        'step_candidate_p95_abs_delta_mu_mumu': float(D64.loc[np.isclose(D64['D'], 6.4), 'bandpass_p95_abs_delta_mu_mumu'].iloc[0]),
        'step_baseline_acceptance_mismatch': float(D64.loc[np.isclose(D64['D'], 6.4), 'width_only_acceptance_mismatch'].iloc[0]),
        'step_candidate_acceptance_mismatch': float(D64.loc[np.isclose(D64['D'], 6.4), 'bandpass_acceptance_mismatch'].iloc[0]),
    },
    {
        'D': 7.2,
        'mechanism': 'tail width gate',
        'parameter': 'beta=0.35',
        'step_baseline_p95_abs_delta_mu_mumu': float(D72['baseline_p95_abs_delta_mu_mumu']),
        'step_candidate_p95_abs_delta_mu_mumu': float(D72['candidate_p95_abs_delta_mu_mumu']),
        'step_baseline_acceptance_mismatch': float(D72['baseline_acceptance_mismatch']),
        'step_candidate_acceptance_mismatch': float(D72['candidate_acceptance_mismatch']),
    },
    {
        'D': 8.0,
        'mechanism': 'tail width gate',
        'parameter': 'beta=1.2',
        'step_baseline_p95_abs_delta_mu_mumu': float(D80['baseline_p95_abs_delta_mu_mumu']),
        'step_candidate_p95_abs_delta_mu_mumu': float(D80['candidate_p95_abs_delta_mu_mumu']),
        'step_baseline_acceptance_mismatch': float(D80['baseline_acceptance_mismatch']),
        'step_candidate_acceptance_mismatch': float(D80['candidate_acceptance_mismatch']),
    },
]
summary = pd.DataFrame(rows).sort_values('D').reset_index(drop=True)
summary = summary.merge(
    COMBINED[['D', 'candidate_p95_abs_delta_mu_mumu', 'candidate_max_abs_delta_mu_mumu', 'candidate_acceptance_mismatch']].rename(
        columns={
            'candidate_p95_abs_delta_mu_mumu': 'combined_p95_abs_delta_mu_mumu',
            'candidate_max_abs_delta_mu_mumu': 'combined_max_abs_delta_mu_mumu',
            'candidate_acceptance_mismatch': 'combined_acceptance_mismatch',
        }
    ),
    on='D',
    how='left',
)
summary.to_csv(OUT_SUMMARY, index=False)

x = np.arange(len(summary))
labels = [f"D={d:.1f}" for d in summary['D']]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(x - 0.18, summary['step_baseline_p95_abs_delta_mu_mumu'], width=0.36, label='before local fix', color='#b07aa1')
axes[0].bar(x + 0.18, summary['step_candidate_p95_abs_delta_mu_mumu'], width=0.36, label='after local fix', color='#4e79a7')
axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
axes[0].set_title('Stepwise p95 |Δmu_mumu| closure')
axes[0].set_ylabel('p95')
axes[0].set_xticks(x, labels)
axes[0].grid(alpha=0.25, axis='y')
axes[0].legend(fontsize=8)

axes[1].bar(x - 0.18, summary['step_baseline_acceptance_mismatch'], width=0.36, label='before local fix', color='#b07aa1')
axes[1].bar(x + 0.18, summary['step_candidate_acceptance_mismatch'], width=0.36, label='after local fix', color='#59a14f')
axes[1].set_title('Stepwise acceptance mismatch closure')
axes[1].set_ylabel('fraction')
axes[1].set_xticks(x, labels)
axes[1].grid(alpha=0.25, axis='y')
axes[1].legend(fontsize=8)

fig.suptitle('Strict all-direct local-fix summary')
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
plt.close(fig)

OUT_META.write_text(json.dumps({'rows': summary.to_dict(orient='records')}, indent=2))
for path in [OUT_SUMMARY, OUT_PNG, OUT_META]:
    (PAPERDIR / path.name).write_bytes(path.read_bytes())

print(f'[saved] {OUT_SUMMARY}')
print(summary.to_dict(orient='records'))

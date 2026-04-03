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
OUT_SUMMARY = OUTDIR / 'runtime_direct_detlin_focusband_generalization_summary.csv'
OUT_DETAIL = OUTDIR / 'runtime_direct_detlin_focusband_generalization_detail.csv'
OUT_PNG = OUTDIR / 'runtime_direct_detlin_focusband_generalization.png'
OUT_META = OUTDIR / 'runtime_direct_detlin_focusband_generalization_run_meta.json'
OUT_DECISION = OUTDIR / 'runtime_direct_detlin_focusband_generalization_decision.json'

TARGETS = [4.0, 4.8, 6.4, 7.2, 8.0]
selected_cases = list(mod.ALL_CASES)
obs = mod.load_observations()['mumu']
mu_obs = float(obs.mu_obs)
sigma_obs = max(float(obs.sigma_obs), 1e-12)

rows = []
for case in selected_cases:
    df = mod._load_map(Path(case['map_path']))
    d_grid = sorted(df['D'].unique())
    eta_grid = sorted(df['eta'].unique())
    kin = mod._build_kinetics(min(d_grid), max(d_grid), len(d_grid))
    mod._patch_gnorm_bandpass(kin)
    mod._patch_width_bands(kin)
    slice_ds = sorted({float(d_grid[int(np.argmin(np.abs(np.asarray(d_grid) - t)))]) for t in TARGETS})
    for D in slice_ds:
        sub = df[np.isclose(df['D'], D)].copy()
        for row in sub.itertuples(index=False):
            mu = float(
                kin.hll_mu_pred(
                    2,
                    D=float(row.D),
                    eta=float(row.eta),
                    t_coh=mod.T_COH,
                    ref_D=mod.REF_D,
                    ref_eta=mod.REF_ETA,
                    observable_mode='eft_wilson_uv_rge',
                    N_max=kin.params.hll_observable_nmax,
                )
            )
            chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
            rows.append({
                'case': case['case'],
                'case_label': case['label'],
                'target_D': float(min(TARGETS, key=lambda t: abs(t - D))),
                'eval_D': float(D),
                'eta': float(row.eta),
                'abs_delta_mu_mumu': float(abs(mu - float(row.mu_mumu))),
                'acceptance_mismatch': float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
            })

detail = pd.DataFrame(rows)
summary = detail.groupby(['case','case_label','target_D','eval_D']).agg(
    n_points=('eta','size'),
    acceptance_mismatch=('acceptance_mismatch','mean'),
    p95_abs_delta_mu_mumu=('abs_delta_mu_mumu', lambda s: float(np.percentile(s,95.0))),
    max_abs_delta_mu_mumu=('abs_delta_mu_mumu','max'),
).reset_index()
case_summary = detail.groupby(['case','case_label']).agg(
    n_points=('eta','size'),
    acceptance_mismatch=('acceptance_mismatch','mean'),
    p95_abs_delta_mu_mumu=('abs_delta_mu_mumu', lambda s: float(np.percentile(s,95.0))),
    max_abs_delta_mu_mumu=('abs_delta_mu_mumu','max'),
).reset_index()
summary.to_csv(OUT_DETAIL, index=False)
case_summary.to_csv(OUT_SUMMARY, index=False)

x = np.arange(len(case_summary))
labels = case_summary['case_label'].tolist()
fig, axes = plt.subplots(1,2, figsize=(11,4.2))
axes[0].bar(x, case_summary['p95_abs_delta_mu_mumu'], color='#4e79a7')
axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
axes[0].set_title('Focus-band slice p95 |Δmu|')
axes[0].set_xticks(x, labels, rotation=25, ha='right')
axes[0].grid(alpha=0.25, axis='y')
axes[1].bar(x, case_summary['acceptance_mismatch'], color='#59a14f')
axes[1].set_title('Focus-band slice acceptance mismatch')
axes[1].set_xticks(x, labels, rotation=25, ha='right')
axes[1].grid(alpha=0.25, axis='y')
fig.suptitle('Strict all-direct focus-band generalization slices')
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
plt.close(fig)

decision = {
    'all_cases_zero_mismatch': bool((case_summary['acceptance_mismatch'] == 0.0).all()),
    'worst_case_p95_abs_delta_mu_mumu': float(case_summary['p95_abs_delta_mu_mumu'].max()),
    'worst_case': str(case_summary.loc[case_summary['p95_abs_delta_mu_mumu'].idxmax(), 'case']),
}
OUT_META.write_text(json.dumps({'cases': selected_cases, 'targets': TARGETS}, indent=2))
OUT_DECISION.write_text(json.dumps(decision, indent=2))
for path in [OUT_SUMMARY, OUT_DETAIL, OUT_PNG, OUT_META, OUT_DECISION]:
    (PAPERDIR / path.name).write_bytes(path.read_bytes())
print(f'[saved] {OUT_SUMMARY}')
print(json.dumps(decision, indent=2))

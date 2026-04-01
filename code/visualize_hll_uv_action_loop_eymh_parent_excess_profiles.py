from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/Users/boypatrick/codex/PSLT_quantam')
OUTDIR = ROOT / 'output' / 'hll_uv_matching'
PAPER = ROOT / 'paper'
MAP = OUTDIR / 'hll_uv_to_eft_map_uv_action_loop_eymh_parented_D41E21_targetdense.csv'
TAG = 'D41E21_targetdense'
LOW_CUT = 9.6
HIGH_SPLIT = 16.8
SELECT_ETA = [0.20, 1.15, 2.10, 3.05, 4.00]


def pivot_metric(df: pd.DataFrame, col: str):
    p = df.pivot(index='eta', columns='D', values=col).sort_index().sort_index(axis=1)
    return p.columns.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float)


def nearest(vals: np.ndarray, target: float) -> float:
    vals = np.asarray(vals, dtype=float)
    return float(vals[np.argmin(np.abs(vals - target))])


def q(x: np.ndarray, quant: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), quant))


def main() -> None:
    df = pd.read_csv(MAP)
    coeff_l1 = np.maximum(df['coeff_l1'].to_numpy(float), 1e-30)
    coeff_l2 = np.maximum(df['coeff_l2'].to_numpy(float), 1e-30)
    gap_cv = np.maximum(df['gap_cv'].to_numpy(float), 0.0)
    c_tree = np.maximum(df['c_tree_diag_cv'].to_numpy(float), 0.0)
    k_part = coeff_l1 / coeff_l2
    k_bg = 1.0 + gap_cv
    k_full = k_bg + c_tree
    g_schur = k_full / np.maximum(k_bg, 1e-30)
    e_part = np.maximum(k_part - 1.0, 0.0)
    e_tree = np.maximum(k_full - k_bg, 0.0)
    xi_target = np.maximum(e_part * e_tree, 0.0) / np.maximum(k_bg * k_part * g_schur, 1e-30)

    plot_df = df[['D', 'eta']].copy()
    plot_df['E_part'] = e_part
    plot_df['E_tree'] = e_tree
    plot_df['xi_target'] = xi_target
    plot_df['k_bg'] = k_bg
    plot_df['k_part'] = k_part
    plot_df['g_schur'] = g_schur

    dvals = np.sort(plot_df['D'].unique())
    eta_vals = np.sort(plot_df['eta'].unique())

    profiles = []
    for eta_t in SELECT_ETA:
        eta = nearest(eta_vals, eta_t)
        sub = plot_df[np.isclose(plot_df['eta'], eta)].sort_values('D')
        for row in sub.itertuples(index=False):
            profiles.append({
                'eta': float(row.eta),
                'D': float(row.D),
                'E_part': float(row.E_part),
                'E_tree': float(row.E_tree),
                'xi_target': float(row.xi_target),
                'E_part_norm': float(row.E_part / max(sub['E_part'].max(), 1e-30)),
                'E_tree_norm': float(row.E_tree / max(sub['E_tree'].max(), 1e-30)),
                'xi_target_norm': float(row.xi_target / max(sub['xi_target'].max(), 1e-30)),
            })
    prof_df = pd.DataFrame(profiles)
    prof_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_profiles_{TAG}.csv'
    prof_df.to_csv(prof_path, index=False)
    prof_df.to_csv(PAPER / prof_path.name, index=False)

    byD = plot_df.groupby('D', as_index=False).agg(
        E_part_med=('E_part', 'median'),
        E_tree_med=('E_tree', 'median'),
        xi_target_med=('xi_target', 'median'),
        E_part_p95=('E_part', lambda s: q(s, 0.95)),
        E_tree_p95=('E_tree', lambda s: q(s, 0.95)),
        xi_target_p95=('xi_target', lambda s: q(s, 0.95)),
        E_part_mean=('E_part', 'mean'),
        E_tree_mean=('E_tree', 'mean'),
        xi_target_mean=('xi_target', 'mean'),
    )
    byD['E_part_norm'] = byD['E_part_med'] / max(float(byD['E_part_med'].max()), 1e-30)
    byD['E_tree_norm'] = byD['E_tree_med'] / max(float(byD['E_tree_med'].max()), 1e-30)
    byD['xi_target_norm'] = byD['xi_target_med'] / max(float(byD['xi_target_med'].max()), 1e-30)
    byD['region'] = np.where(byD['D'] <= LOW_CUT, 'low-D', np.where(byD['D'] <= HIGH_SPLIT, 'transition', 'high-D'))
    byD_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_byD_{TAG}.csv'
    byD.to_csv(byD_path, index=False)
    byD.to_csv(PAPER / byD_path.name, index=False)

    summary = pd.DataFrame([
        {
            'region': 'full',
            'D_min': float(plot_df['D'].min()),
            'D_max': float(plot_df['D'].max()),
            'peak_D_E_part': float(byD.loc[byD['E_part_med'].idxmax(), 'D']),
            'peak_D_E_tree': float(byD.loc[byD['E_tree_med'].idxmax(), 'D']),
            'peak_D_xi_target': float(byD.loc[byD['xi_target_med'].idxmax(), 'D']),
            'corr_Epart_xi': float(np.corrcoef(plot_df['E_part'], plot_df['xi_target'])[0,1]),
            'corr_Etree_xi': float(np.corrcoef(plot_df['E_tree'], plot_df['xi_target'])[0,1]),
            'eta_var_frac_E_part': float(plot_df.groupby('D')['E_part'].std().fillna(0).mean() / max(plot_df.groupby('D')['E_part'].mean().mean(),1e-30)),
            'eta_var_frac_E_tree': float(plot_df.groupby('D')['E_tree'].std().fillna(0).mean() / max(plot_df.groupby('D')['E_tree'].mean().mean(),1e-30)),
            'eta_var_frac_xi': float(plot_df.groupby('D')['xi_target'].std().fillna(0).mean() / max(plot_df.groupby('D')['xi_target'].mean().mean(),1e-30)),
        }
    ] + [
        {
            'region': region,
            'D_min': float(sub['D'].min()),
            'D_max': float(sub['D'].max()),
            'peak_D_E_part': float(sub.loc[sub['E_part_med'].idxmax(), 'D']),
            'peak_D_E_tree': float(sub.loc[sub['E_tree_med'].idxmax(), 'D']),
            'peak_D_xi_target': float(sub.loc[sub['xi_target_med'].idxmax(), 'D']),
            'corr_Epart_xi': float(np.corrcoef(sub['E_part_med'], sub['xi_target_med'])[0,1]) if len(sub) > 1 else np.nan,
            'corr_Etree_xi': float(np.corrcoef(sub['E_tree_med'], sub['xi_target_med'])[0,1]) if len(sub) > 1 else np.nan,
            'eta_var_frac_E_part': float(plot_df[plot_df['D'].isin(sub['D'])].groupby('D')['E_part'].std().fillna(0).mean() / max(plot_df[plot_df['D'].isin(sub['D'])].groupby('D')['E_part'].mean().mean(),1e-30)),
            'eta_var_frac_E_tree': float(plot_df[plot_df['D'].isin(sub['D'])].groupby('D')['E_tree'].std().fillna(0).mean() / max(plot_df[plot_df['D'].isin(sub['D'])].groupby('D')['E_tree'].mean().mean(),1e-30)),
            'eta_var_frac_xi': float(plot_df[plot_df['D'].isin(sub['D'])].groupby('D')['xi_target'].std().fillna(0).mean() / max(plot_df[plot_df['D'].isin(sub['D'])].groupby('D')['xi_target'].mean().mean(),1e-30)),
        }
        for region, sub in byD.groupby('region')
    ])
    summary_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_summary_{TAG}.csv'
    summary.to_csv(summary_path, index=False)
    summary.to_csv(PAPER / summary_path.name, index=False)

    Dg, Eg, Epart_grid = pivot_metric(plot_df, 'E_part')
    _, _, Etree_grid = pivot_metric(plot_df, 'E_tree')
    _, _, Xgrid = pivot_metric(plot_df, 'xi_target')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    im0 = axes[0,0].imshow(Epart_grid, aspect='auto', origin='lower', extent=[Dg.min(), Dg.max(), Eg.min(), Eg.max()])
    axes[0,0].set_title(r'$E_{\rm part}(D,\eta)$')
    axes[0,0].set_xlabel('D')
    axes[0,0].set_ylabel(r'$\eta$')
    axes[0,0].axvline(LOW_CUT, color='w', ls='--', lw=1)
    axes[0,0].axvline(HIGH_SPLIT, color='w', ls=':', lw=1)
    fig.colorbar(im0, ax=axes[0,0], shrink=0.85)

    im1 = axes[0,1].imshow(Etree_grid, aspect='auto', origin='lower', extent=[Dg.min(), Dg.max(), Eg.min(), Eg.max()])
    axes[0,1].set_title(r'$E_{\rm tree}(D,\eta)$')
    axes[0,1].set_xlabel('D')
    axes[0,1].set_ylabel(r'$\eta$')
    axes[0,1].axvline(LOW_CUT, color='w', ls='--', lw=1)
    axes[0,1].axvline(HIGH_SPLIT, color='w', ls=':', lw=1)
    fig.colorbar(im1, ax=axes[0,1], shrink=0.85)

    ax = axes[1,0]
    ax.plot(byD['D'], byD['E_part_norm'], label=r'median $E_{\rm part}$', lw=2)
    ax.plot(byD['D'], byD['E_tree_norm'], label=r'median $E_{\rm tree}$', lw=2)
    ax.plot(byD['D'], byD['xi_target_norm'], label=r'median $\xi_{\rm target}$', lw=2.2, color='k')
    ax.axvspan(byD['D'].min(), LOW_CUT, color='#d8ecff', alpha=0.25)
    ax.axvspan(LOW_CUT, HIGH_SPLIT, color='#fff0c9', alpha=0.25)
    ax.axvspan(HIGH_SPLIT, byD['D'].max(), color='#f2e1ff', alpha=0.25)
    ax.axvline(LOW_CUT, color='0.35', ls='--', lw=1)
    ax.axvline(HIGH_SPLIT, color='0.35', ls=':', lw=1)
    ax.set_title('Median normalized dense-D profiles')
    ax.set_xlabel('D')
    ax.set_ylabel('normalized amplitude')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1,1]
    for eta_t in SELECT_ETA:
        eta = nearest(eta_vals, eta_t)
        sub = prof_df[np.isclose(prof_df['eta'], eta)].sort_values('D')
        ax.plot(sub['D'], sub['E_part_norm'], lw=1.1, alpha=0.8, label=fr'$E_{{\rm part}},\eta={eta:.2f}$')
        ax.plot(sub['D'], sub['E_tree_norm'], lw=1.1, alpha=0.8, ls='--', label=fr'$E_{{\rm tree}},\eta={eta:.2f}$')
    ax.axvline(LOW_CUT, color='0.35', ls='--', lw=1)
    ax.axvline(HIGH_SPLIT, color='0.35', ls=':', lw=1)
    ax.set_title('Selected $\eta$ slices: normalized excess profiles')
    ax.set_xlabel('D')
    ax.set_ylabel('normalized amplitude')
    ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle('EYMH parented dense-D excess decomposition: who drives oscillation vs cutoff', fontsize=15)

    fig_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_profiles_{TAG}.png'
    fig.savefig(fig_path, dpi=220)
    fig.savefig(PAPER / fig_path.name, dpi=220)
    plt.close(fig)

    meta = {
        'input_map': str(MAP),
        'note': 'Dense-D decomposition of parented EYMH target into E_part and E_tree.',
        'low_cut': LOW_CUT,
        'high_split': HIGH_SPLIT,
        'selected_eta': SELECT_ETA,
    }
    meta_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_profiles_run_meta_{TAG}.json'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (PAPER / meta_path.name).write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(fig_path)
    print(summary_path)
    print(byD_path)
    print(prof_path)

if __name__ == '__main__':
    main()

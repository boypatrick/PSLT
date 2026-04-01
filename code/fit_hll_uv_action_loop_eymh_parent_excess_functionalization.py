from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path('/Users/boypatrick/codex/PSLT_quantam')
OUTDIR = ROOT / 'output' / 'hll_uv_matching'
PAPER = ROOT / 'paper'
MAP = OUTDIR / 'hll_uv_to_eft_map_uv_action_loop_eymh_parented_D41E21_targetdense.csv'
TAG = 'D41E21_targetdense'
LOW_CUT = 9.6
HIGH_SPLIT = 16.8


def logistic(d: np.ndarray, k: float, dc: float) -> np.ndarray:
    x = np.clip(k * (d - dc), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(x))


def env_exp(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.exp(np.clip(a + b * d + c * d * d, -50.0, 50.0))


def model_smooth(d: np.ndarray, a: float, b: float, c: float, k: float, dc: float) -> np.ndarray:
    return env_exp(d, a, b, c) * logistic(d, k, dc)


def model_osc(d: np.ndarray, a: float, b: float, c: float, A: float, omega: float, phi: float, k: float, dc: float) -> np.ndarray:
    osc = 1.0 + A * np.sin(omega * d + phi)
    return env_exp(d, a, b, c) * np.maximum(osc, 1e-6) * logistic(d, k, dc)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def q(x: np.ndarray, quant: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), quant))


def fit_curve(fn, d, y, p0, bounds):
    popt, _ = curve_fit(fn, d, y, p0=p0, bounds=bounds, maxfev=50000)
    pred = fn(d, *popt)
    return popt, pred


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
    xi_target = np.maximum(e_part * e_tree, 0.0) / np.maximum(k_part * k_full, 1e-30)

    work = pd.DataFrame({'D': df['D'], 'eta': df['eta'], 'k_bg': k_bg, 'E_part': e_part, 'E_tree': e_tree, 'xi_target': xi_target})
    byD = work.groupby('D', as_index=False).agg(
        E_part=('E_part', 'median'),
        E_tree=('E_tree', 'median'),
        xi_target=('xi_target', 'median'),
        k_bg=('k_bg', 'median'),
    )
    d = byD['D'].to_numpy(float)
    y_part = byD['E_part'].to_numpy(float)
    y_tree = byD['E_tree'].to_numpy(float)
    y_xi = byD['xi_target'].to_numpy(float)
    kbg = byD['k_bg'].to_numpy(float)
    support_mask = d < 19.2

    # Fits
    part_s_popt, part_s_pred = fit_curve(
        model_smooth, d, y_part,
        p0=[-1.0, 0.04, -0.0005, 25.0, 19.0],
        bounds=([-8.0, -1.0, -0.1, 1.0, 17.0], [4.0, 1.0, 0.1, 200.0, 21.0]),
    )
    part_o_popt, part_o_pred = fit_curve(
        model_osc, d, y_part,
        p0=[-1.0, 0.04, -0.0005, 0.05, 2.0, 0.0, 25.0, 19.0],
        bounds=([-8.0, -1.0, -0.1, 0.0, 0.2, -np.pi, 1.0, 17.0], [4.0, 1.0, 0.1, 0.95, 5.0, np.pi, 200.0, 21.0]),
    )
    tree_s_popt, tree_s_pred = fit_curve(
        model_smooth, d, y_tree,
        p0=[0.0, -0.02, -0.001, 25.0, 19.0],
        bounds=([-8.0, -1.0, -0.1, 1.0, 17.0], [4.0, 1.0, 0.1, 200.0, 21.0]),
    )
    tree_o_popt, tree_o_pred = fit_curve(
        model_osc, d, y_tree,
        p0=[0.0, -0.02, -0.001, 0.25, 2.0, 0.0, 25.0, 19.0],
        bounds=([-8.0, -1.0, -0.1, 0.0, 0.2, -np.pi, 1.0, 17.0], [4.0, 1.0, 0.1, 0.95, 5.0, np.pi, 200.0, 21.0]),
    )
    tree_s_refit_support_popt, tree_s_refit_support_pred = fit_curve(
        model_smooth, d[support_mask], y_tree[support_mask],
        p0=[0.0, -0.02, -0.001, 25.0, 19.0],
        bounds=([-8.0, -1.0, -0.1, 1.0, 17.0], [4.0, 1.0, 0.1, 200.0, 21.0]),
    )
    tree_o_refit_support_popt, tree_o_refit_support_pred = fit_curve(
        model_osc, d[support_mask], y_tree[support_mask],
        p0=[0.0, -0.02, -0.001, 0.25, 2.0, 0.0, 25.0, 19.0],
        bounds=([-8.0, -1.0, -0.1, 0.0, 0.2, -np.pi, 1.0, 17.0], [4.0, 1.0, 0.1, 0.95, 5.0, np.pi, 200.0, 21.0]),
    )

    xi_part_s_tree_o = (part_s_pred * tree_o_pred) / np.maximum((1.0 + part_s_pred) * (kbg + tree_o_pred), 1e-30)
    xi_part_o_tree_o = (part_o_pred * tree_o_pred) / np.maximum((1.0 + part_o_pred) * (kbg + tree_o_pred), 1e-30)
    xi_part_s_tree_s = (part_s_pred * tree_s_pred) / np.maximum((1.0 + part_s_pred) * (kbg + tree_s_pred), 1e-30)

    summary_rows = [
        {
            'component': 'E_part_smooth',
            'rmse': rmse(y_part, part_s_pred),
            'p95_abs_resid': q(np.abs(y_part - part_s_pred), 0.95),
            'a': part_s_popt[0], 'b': part_s_popt[1], 'c': part_s_popt[2], 'k': part_s_popt[3], 'Dc': part_s_popt[4],
        },
        {
            'component': 'E_part_osc',
            'rmse': rmse(y_part, part_o_pred),
            'p95_abs_resid': q(np.abs(y_part - part_o_pred), 0.95),
            'a': part_o_popt[0], 'b': part_o_popt[1], 'c': part_o_popt[2], 'A': part_o_popt[3], 'omega': part_o_popt[4], 'phi': part_o_popt[5], 'k': part_o_popt[6], 'Dc': part_o_popt[7],
        },
        {
            'component': 'E_tree_smooth',
            'rmse': rmse(y_tree, tree_s_pred),
            'p95_abs_resid': q(np.abs(y_tree - tree_s_pred), 0.95),
            'a': tree_s_popt[0], 'b': tree_s_popt[1], 'c': tree_s_popt[2], 'k': tree_s_popt[3], 'Dc': tree_s_popt[4],
        },
        {
            'component': 'E_tree_osc',
            'rmse': rmse(y_tree, tree_o_pred),
            'p95_abs_resid': q(np.abs(y_tree - tree_o_pred), 0.95),
            'a': tree_o_popt[0], 'b': tree_o_popt[1], 'c': tree_o_popt[2], 'A': tree_o_popt[3], 'omega': tree_o_popt[4], 'phi': tree_o_popt[5], 'k': tree_o_popt[6], 'Dc': tree_o_popt[7],
        },
        {
            'component': 'E_tree_smooth_support_only',
            'rmse': rmse(y_tree[support_mask], tree_s_pred[support_mask]),
            'p95_abs_resid': q(np.abs(y_tree[support_mask] - tree_s_pred[support_mask]), 0.95),
            'a': tree_s_popt[0], 'b': tree_s_popt[1], 'c': tree_s_popt[2], 'k': tree_s_popt[3], 'Dc': tree_s_popt[4],
        },
        {
            'component': 'E_tree_osc_support_only',
            'rmse': rmse(y_tree[support_mask], tree_o_pred[support_mask]),
            'p95_abs_resid': q(np.abs(y_tree[support_mask] - tree_o_pred[support_mask]), 0.95),
            'a': tree_o_popt[0], 'b': tree_o_popt[1], 'c': tree_o_popt[2], 'A': tree_o_popt[3], 'omega': tree_o_popt[4], 'phi': tree_o_popt[5], 'k': tree_o_popt[6], 'Dc': tree_o_popt[7],
        },
        {
            'component': 'E_tree_smooth_refit_support_only',
            'rmse': rmse(y_tree[support_mask], tree_s_refit_support_pred),
            'p95_abs_resid': q(np.abs(y_tree[support_mask] - tree_s_refit_support_pred), 0.95),
            'a': tree_s_refit_support_popt[0], 'b': tree_s_refit_support_popt[1], 'c': tree_s_refit_support_popt[2], 'k': tree_s_refit_support_popt[3], 'Dc': tree_s_refit_support_popt[4],
        },
        {
            'component': 'E_tree_osc_refit_support_only',
            'rmse': rmse(y_tree[support_mask], tree_o_refit_support_pred),
            'p95_abs_resid': q(np.abs(y_tree[support_mask] - tree_o_refit_support_pred), 0.95),
            'a': tree_o_refit_support_popt[0], 'b': tree_o_refit_support_popt[1], 'c': tree_o_refit_support_popt[2], 'A': tree_o_refit_support_popt[3], 'omega': tree_o_refit_support_popt[4], 'phi': tree_o_refit_support_popt[5], 'k': tree_o_refit_support_popt[6], 'Dc': tree_o_refit_support_popt[7],
        },
        {
            'component': 'xi_recon_part_smooth_tree_smooth',
            'rmse': rmse(y_xi, xi_part_s_tree_s),
            'p95_abs_resid': q(np.abs(y_xi - xi_part_s_tree_s), 0.95),
        },
        {
            'component': 'xi_recon_part_smooth_tree_osc',
            'rmse': rmse(y_xi, xi_part_s_tree_o),
            'p95_abs_resid': q(np.abs(y_xi - xi_part_s_tree_o), 0.95),
        },
        {
            'component': 'xi_recon_part_osc_tree_osc',
            'rmse': rmse(y_xi, xi_part_o_tree_o),
            'p95_abs_resid': q(np.abs(y_xi - xi_part_o_tree_o), 0.95),
        },
        {
            'component': 'xi_recon_part_smooth_tree_smooth_support_only',
            'rmse': rmse(y_xi[support_mask], xi_part_s_tree_s[support_mask]),
            'p95_abs_resid': q(np.abs(y_xi[support_mask] - xi_part_s_tree_s[support_mask]), 0.95),
        },
        {
            'component': 'xi_recon_part_smooth_tree_osc_support_only',
            'rmse': rmse(y_xi[support_mask], xi_part_s_tree_o[support_mask]),
            'p95_abs_resid': q(np.abs(y_xi[support_mask] - xi_part_s_tree_o[support_mask]), 0.95),
        },
        {
            'component': 'xi_recon_part_osc_tree_osc_support_only',
            'rmse': rmse(y_xi[support_mask], xi_part_o_tree_o[support_mask]),
            'p95_abs_resid': q(np.abs(y_xi[support_mask] - xi_part_o_tree_o[support_mask]), 0.95),
        },
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_fit_summary_{TAG}.csv'
    summary.to_csv(summary_path, index=False)
    summary.to_csv(PAPER / summary_path.name, index=False)

    fit_df = pd.DataFrame({
        'D': d,
        'k_bg_med': kbg,
        'E_part_med': y_part,
        'E_part_smooth_fit': part_s_pred,
        'E_part_osc_fit': part_o_pred,
        'E_tree_med': y_tree,
        'E_tree_smooth_fit': tree_s_pred,
        'E_tree_osc_fit': tree_o_pred,
        'xi_target_med': y_xi,
        'xi_recon_part_smooth_tree_smooth': xi_part_s_tree_s,
        'xi_recon_part_smooth_tree_osc': xi_part_s_tree_o,
        'xi_recon_part_osc_tree_osc': xi_part_o_tree_o,
    })
    fit_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_fit_profiles_{TAG}.csv'
    fit_df.to_csv(fit_path, index=False)
    fit_df.to_csv(PAPER / fit_path.name, index=False)

    decision = {
        'input_map': str(MAP),
        'interpretation': 'Test whether E_part needs oscillation and whether E_tree carries the oscillation/cutoff structure.',
        'best_xi_reconstruction': 'part_smooth_tree_osc' if rmse(y_xi, xi_part_s_tree_o) <= min(rmse(y_xi, xi_part_s_tree_s), rmse(y_xi, xi_part_o_tree_o)) else 'other',
        'part_osc_gain_over_smooth': rmse(y_part, part_s_pred) - rmse(y_part, part_o_pred),
        'tree_osc_gain_over_smooth': rmse(y_tree, tree_s_pred) - rmse(y_tree, tree_o_pred),
        'xi_gain_tree_osc_over_all_smooth': rmse(y_xi, xi_part_s_tree_s) - rmse(y_xi, xi_part_s_tree_o),
        'tree_osc_gain_over_smooth_support_only': rmse(y_tree[support_mask], tree_s_pred[support_mask]) - rmse(y_tree[support_mask], tree_o_pred[support_mask]),
        'xi_gain_tree_osc_over_all_smooth_support_only': rmse(y_xi[support_mask], xi_part_s_tree_s[support_mask]) - rmse(y_xi[support_mask], xi_part_s_tree_o[support_mask]),
        'tree_osc_refit_gain_over_smooth_refit_support_only': rmse(y_tree[support_mask], tree_s_refit_support_pred) - rmse(y_tree[support_mask], tree_o_refit_support_pred),
    }
    decision_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_fit_decision_{TAG}.json'
    decision_path.write_text(json.dumps(decision, indent=2), encoding='utf-8')
    (PAPER / decision_path.name).write_text(json.dumps(decision, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0,0]
    ax.plot(d, y_part, 'o-', label='E_part data', color='#1f4e79')
    ax.plot(d, part_s_pred, label='smooth fit', color='#2a9d8f', lw=2)
    ax.plot(d, part_o_pred, label='osc fit', color='#e76f51', lw=1.8, ls='--')
    ax.axvline(LOW_CUT, color='0.5', ls='--', lw=1)
    ax.axvline(HIGH_SPLIT, color='0.5', ls=':', lw=1)
    ax.set_title('E_part: smooth vs oscillatory fit')
    ax.set_xlabel('D')
    ax.set_ylabel('E_part')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0,1]
    ax.plot(d, y_tree, 'o-', label='E_tree data', color='#6a1b9a')
    ax.plot(d, tree_s_pred, label='smooth fit', color='#2a9d8f', lw=2)
    ax.plot(d, tree_o_pred, label='osc fit', color='#e76f51', lw=1.8, ls='--')
    ax.axvline(LOW_CUT, color='0.5', ls='--', lw=1)
    ax.axvline(HIGH_SPLIT, color='0.5', ls=':', lw=1)
    ax.set_title('E_tree: smooth vs oscillatory fit')
    ax.set_xlabel('D')
    ax.set_ylabel('E_tree')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1,0]
    ax.plot(d, y_xi, 'o-', label='xi_target data', color='k')
    ax.plot(d, xi_part_s_tree_s, label='part smooth × tree smooth', color='#577590', lw=1.8)
    ax.plot(d, xi_part_s_tree_o, label='part smooth × tree osc', color='#f3722c', lw=2.2)
    ax.plot(d, xi_part_o_tree_o, label='part osc × tree osc', color='#90be6d', lw=1.6, ls='--')
    ax.axvline(LOW_CUT, color='0.5', ls='--', lw=1)
    ax.axvline(HIGH_SPLIT, color='0.5', ls=':', lw=1)
    ax.set_title('xi reconstruction from fitted excess sectors')
    ax.set_xlabel('D')
    ax.set_ylabel('xi_target')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1,1]
    ax.plot(d, kbg, color='#4d908e', lw=2, label='k_bg median')
    ax.plot(d, np.abs(y_xi - xi_part_s_tree_o), color='#f94144', lw=2, label='|xi resid|: part smooth × tree osc')
    ax.axvline(LOW_CUT, color='0.5', ls='--', lw=1)
    ax.axvline(HIGH_SPLIT, color='0.5', ls=':', lw=1)
    ax.set_title('Background normalization and best xi residual')
    ax.set_xlabel('D')
    ax.set_ylabel('value')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle('EYMH dense-D excess functionalization: smooth E_part vs oscillatory E_tree', fontsize=15)
    fig_path = OUTDIR / f'hll_uv_action_loop_eymh_parent_excess_fit_{TAG}.png'
    fig.savefig(fig_path, dpi=220)
    fig.savefig(PAPER / fig_path.name, dpi=220)
    plt.close(fig)

    print(fig_path)
    print(summary_path)
    print(fit_path)
    print(decision_path)

if __name__ == '__main__':
    main()

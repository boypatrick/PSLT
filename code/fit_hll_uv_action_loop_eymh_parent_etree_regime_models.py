from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path("/Users/boypatrick/codex/PSLT_quantam")
OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPER = ROOT / "paper"
MAP = OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D41E21_targetdense.csv"
TAG = "D41E21_targetdense"


def logistic(d: np.ndarray, k: float, dc: float) -> np.ndarray:
    x = np.clip(k * (d - dc), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(x))


def env_exp(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.exp(np.clip(a + b * d + c * d * d, -50.0, 50.0))


def model_smooth(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, b, c, k, dc = p
    return env_exp(d, a, b, c) * logistic(d, k, dc)


def model_twofreq(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, b, c, A1, w1, ph1, A2, w2, ph2, k, dc = p
    osc = 1.0 + A1 * np.sin(w1 * d + ph1) + A2 * np.sin(w2 * d + ph2)
    return env_exp(d, a, b, c) * np.maximum(osc, 1e-6) * logistic(d, k, dc)


def model_low_twofreq(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, b, c, A1, w1, ph1, A2, w2, ph2 = p
    osc = 1.0 + A1 * np.sin(w1 * d + ph1) + A2 * np.sin(w2 * d + ph2)
    return env_exp(d, a, b, c) * np.maximum(osc, 1e-6)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def q(x: np.ndarray, quant: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), quant))


def fit_multistart(model_fn, x: np.ndarray, y: np.ndarray, seeds: list[np.ndarray], lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    for seed in seeds:
        try:
            res = least_squares(lambda p: model_fn(x, p) - y, x0=seed, bounds=(lb, ub), max_nfev=50000)
            pred = model_fn(x, res.x)
            score = rmse(y, pred)
            if best is None or score < best[2]:
                best = (res.x.copy(), pred.copy(), score)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("all multistart fits failed")
    return best


def random_seeds(lb: np.ndarray, ub: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [lb + (ub - lb) * rng.random(len(lb)) for _ in range(n)]


def xi_from_parts(e_part: np.ndarray, e_tree: np.ndarray, k_bg: np.ndarray) -> np.ndarray:
    return np.maximum(e_part * e_tree, 0.0) / np.maximum((1.0 + e_part) * (k_bg + e_tree), 1e-30)


def main() -> None:
    rng = np.random.default_rng(0)

    df = pd.read_csv(MAP)
    coeff_l1 = np.maximum(df["coeff_l1"].to_numpy(float), 1e-30)
    coeff_l2 = np.maximum(df["coeff_l2"].to_numpy(float), 1e-30)
    gap_cv = np.maximum(df["gap_cv"].to_numpy(float), 0.0)
    c_tree = np.maximum(df["c_tree_diag_cv"].to_numpy(float), 0.0)

    k_part = coeff_l1 / coeff_l2
    k_bg = 1.0 + gap_cv
    k_full = k_bg + c_tree
    e_part = np.maximum(k_part - 1.0, 0.0)
    e_tree = np.maximum(k_full - k_bg, 0.0)
    xi_target = xi_from_parts(e_part, e_tree, k_bg)

    work = pd.DataFrame({"D": df["D"], "eta": df["eta"], "k_bg": k_bg, "E_part": e_part, "E_tree": e_tree, "xi_target": xi_target})
    byD = work.groupby("D", as_index=False).agg(
        E_part=("E_part", "median"),
        E_tree=("E_tree", "median"),
        xi_target=("xi_target", "median"),
        k_bg=("k_bg", "median"),
    )

    d = byD["D"].to_numpy(float)
    y_part = byD["E_part"].to_numpy(float)
    y_tree = byD["E_tree"].to_numpy(float)
    y_xi = byD["xi_target"].to_numpy(float)
    kbg = byD["k_bg"].to_numpy(float)

    # Smooth E_part baseline
    part_lb = np.array([-8.0, -1.0, -0.1, 1.0, 17.0], dtype=float)
    part_ub = np.array([4.0, 1.0, 0.1, 200.0, 21.0], dtype=float)
    part_seeds = [np.array([-1.2, 0.08, -0.0025, 80.0, 19.0], dtype=float)] + random_seeds(part_lb, part_ub, 24, rng)
    part_p, part_pred, _ = fit_multistart(model_smooth, d, y_part, part_seeds, part_lb, part_ub)

    # Global smooth / two-frequency E_tree
    tree_s_lb = np.array([-8.0, -1.0, -0.1, 1.0, 17.0], dtype=float)
    tree_s_ub = np.array([4.0, 1.0, 0.1, 200.0, 21.0], dtype=float)
    tree_s_seeds = [np.array([0.2, -0.06, 0.002, 10.0, 18.8], dtype=float)] + random_seeds(tree_s_lb, tree_s_ub, 24, rng)
    tree_s_p, tree_s_pred, _ = fit_multistart(model_smooth, d, y_tree, tree_s_seeds, tree_s_lb, tree_s_ub)

    tree_tf_lb = np.array([-8.0, -1.0, -0.1, 0.0, 0.2, -np.pi, 0.0, 0.8, -np.pi, 1.0, 17.0], dtype=float)
    tree_tf_ub = np.array([4.0, 1.0, 0.1, 0.95, 2.2, np.pi, 0.95, 4.5, np.pi, 200.0, 21.0], dtype=float)
    tree_tf_seed0 = np.array([0.2, -0.06, 0.002, 0.10, 1.0, 0.0, 0.12, 1.95, 2.7, 60.0, 18.5], dtype=float)
    tree_tf_seeds = [tree_tf_seed0] + random_seeds(tree_tf_lb, tree_tf_ub, 48, rng)
    tree_tf_p, tree_tf_pred, _ = fit_multistart(model_twofreq, d, y_tree, tree_tf_seeds, tree_tf_lb, tree_tf_ub)

    # Piecewise low/high regime fit
    split_grid = [12.0, 12.8, 13.6, 14.4, 15.2, 16.0, 16.8]
    piece_best = None
    piece_records = []
    low_lb = np.array([-8.0, -1.0, -0.1, 0.0, 0.2, -np.pi, 0.0, 0.8, -np.pi], dtype=float)
    low_ub = np.array([4.0, 1.0, 0.1, 0.95, 2.2, np.pi, 0.95, 4.5, np.pi], dtype=float)
    high_lb_base = np.array([-8.0, -1.0, -0.1, 1.0, 17.0], dtype=float)
    high_ub_base = np.array([4.0, 1.0, 0.1, 200.0, 21.0], dtype=float)

    for split in split_grid:
        low_mask = d <= split
        high_mask = d > split
        if low_mask.sum() < 8 or high_mask.sum() < 4:
            continue

        low_seed0 = np.array([0.0, -0.03, 0.001, 0.10, 1.0, 0.0, 0.10, 1.95, 2.7], dtype=float)
        low_seeds = [low_seed0] + random_seeds(low_lb, low_ub, 36, rng)
        low_p, low_pred_sub, _ = fit_multistart(model_low_twofreq, d[low_mask], y_tree[low_mask], low_seeds, low_lb, low_ub)

        high_lb = high_lb_base.copy()
        high_ub = high_ub_base.copy()
        high_lb[4] = max(split, 17.0)
        high_seed0 = np.array([0.2, -0.05, 0.001, 30.0, max(split + 0.8, 18.2)], dtype=float)
        high_seeds = [high_seed0] + random_seeds(high_lb, high_ub, 24, rng)
        high_p, high_pred_sub, _ = fit_multistart(model_smooth, d[high_mask], y_tree[high_mask], high_seeds, high_lb, high_ub)

        piece_pred = np.empty_like(y_tree)
        piece_pred[low_mask] = model_low_twofreq(d[low_mask], low_p)
        piece_pred[high_mask] = model_smooth(d[high_mask], high_p)
        xi_piece = xi_from_parts(part_pred, piece_pred, kbg)

        rec = {
            "split_D": split,
            "E_tree_piecewise_rmse": rmse(y_tree, piece_pred),
            "E_tree_piecewise_p95_abs_resid": q(np.abs(y_tree - piece_pred), 0.95),
            "xi_piecewise_rmse": rmse(y_xi, xi_piece),
            "xi_piecewise_p95_abs_resid": q(np.abs(y_xi - xi_piece), 0.95),
            "low_A1": low_p[3],
            "low_w1": low_p[4],
            "low_A2": low_p[6],
            "low_w2": low_p[7],
            "high_k": high_p[3],
            "high_Dc": high_p[4],
        }
        piece_records.append(rec)
        score = rec["xi_piecewise_rmse"] + 0.5 * rec["E_tree_piecewise_rmse"]
        if piece_best is None or score < piece_best["score"]:
            piece_best = {"score": score, "split": split, "low_p": low_p, "high_p": high_p, "pred": piece_pred, "xi": xi_piece, "metrics": rec}

    piece_df = pd.DataFrame(piece_records).sort_values("xi_piecewise_rmse")
    piece_path = OUTDIR / f"hll_uv_action_loop_eymh_parent_etree_piecewise_scan_{TAG}.csv"
    piece_df.to_csv(piece_path, index=False)
    piece_df.to_csv(PAPER / piece_path.name, index=False)

    xi_smooth = xi_from_parts(part_pred, tree_s_pred, kbg)
    xi_twofreq = xi_from_parts(part_pred, tree_tf_pred, kbg)

    summary = pd.DataFrame(
        [
            {
                "model": "E_part_smooth",
                "rmse": rmse(y_part, part_pred),
                "p95_abs_resid": q(np.abs(y_part - part_pred), 0.95),
                "a": part_p[0],
                "b": part_p[1],
                "c": part_p[2],
                "k": part_p[3],
                "Dc": part_p[4],
            },
            {
                "model": "E_tree_smooth",
                "rmse": rmse(y_tree, tree_s_pred),
                "p95_abs_resid": q(np.abs(y_tree - tree_s_pred), 0.95),
                "a": tree_s_p[0],
                "b": tree_s_p[1],
                "c": tree_s_p[2],
                "k": tree_s_p[3],
                "Dc": tree_s_p[4],
            },
            {
                "model": "E_tree_twofreq_global",
                "rmse": rmse(y_tree, tree_tf_pred),
                "p95_abs_resid": q(np.abs(y_tree - tree_tf_pred), 0.95),
                "a": tree_tf_p[0],
                "b": tree_tf_p[1],
                "c": tree_tf_p[2],
                "A1": tree_tf_p[3],
                "omega1": tree_tf_p[4],
                "phi1": tree_tf_p[5],
                "A2": tree_tf_p[6],
                "omega2": tree_tf_p[7],
                "phi2": tree_tf_p[8],
                "k": tree_tf_p[9],
                "Dc": tree_tf_p[10],
            },
            {
                "model": "E_tree_piecewise_best",
                "rmse": rmse(y_tree, piece_best["pred"]),
                "p95_abs_resid": q(np.abs(y_tree - piece_best["pred"]), 0.95),
                "split_D": piece_best["split"],
                "low_A1": piece_best["low_p"][3],
                "low_omega1": piece_best["low_p"][4],
                "low_A2": piece_best["low_p"][6],
                "low_omega2": piece_best["low_p"][7],
                "high_k": piece_best["high_p"][3],
                "high_Dc": piece_best["high_p"][4],
            },
            {
                "model": "xi_recon_smooth_tree_smooth",
                "rmse": rmse(y_xi, xi_smooth),
                "p95_abs_resid": q(np.abs(y_xi - xi_smooth), 0.95),
            },
            {
                "model": "xi_recon_smooth_tree_twofreq_global",
                "rmse": rmse(y_xi, xi_twofreq),
                "p95_abs_resid": q(np.abs(y_xi - xi_twofreq), 0.95),
            },
            {
                "model": "xi_recon_smooth_tree_piecewise_best",
                "rmse": rmse(y_xi, piece_best["xi"]),
                "p95_abs_resid": q(np.abs(y_xi - piece_best["xi"]), 0.95),
                "split_D": piece_best["split"],
            },
        ]
    )
    summary_path = OUTDIR / f"hll_uv_action_loop_eymh_parent_etree_regime_fit_summary_{TAG}.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(PAPER / summary_path.name, index=False)

    prof = pd.DataFrame(
        {
            "D": d,
            "k_bg_med": kbg,
            "E_part_med": y_part,
            "E_part_smooth_fit": part_pred,
            "E_tree_med": y_tree,
            "E_tree_smooth_fit": tree_s_pred,
            "E_tree_twofreq_global_fit": tree_tf_pred,
            "E_tree_piecewise_best_fit": piece_best["pred"],
            "xi_target_med": y_xi,
            "xi_recon_smooth_tree_smooth": xi_smooth,
            "xi_recon_smooth_tree_twofreq_global": xi_twofreq,
            "xi_recon_smooth_tree_piecewise_best": piece_best["xi"],
        }
    )
    prof_path = OUTDIR / f"hll_uv_action_loop_eymh_parent_etree_regime_fit_profiles_{TAG}.csv"
    prof.to_csv(prof_path, index=False)
    prof.to_csv(PAPER / prof_path.name, index=False)

    decision = {
        "input_map": str(MAP),
        "best_etree_model": "piecewise" if rmse(y_tree, piece_best["pred"]) <= min(rmse(y_tree, tree_s_pred), rmse(y_tree, tree_tf_pred)) else ("twofreq_global" if rmse(y_tree, tree_tf_pred) < rmse(y_tree, tree_s_pred) else "smooth"),
        "best_xi_reconstruction": "piecewise" if rmse(y_xi, piece_best["xi"]) <= min(rmse(y_xi, xi_smooth), rmse(y_xi, xi_twofreq)) else ("twofreq_global" if rmse(y_xi, xi_twofreq) < rmse(y_xi, xi_smooth) else "smooth"),
        "piecewise_split_D": piece_best["split"],
        "global_twofreq_gain_over_smooth": rmse(y_tree, tree_s_pred) - rmse(y_tree, tree_tf_pred),
        "piecewise_gain_over_smooth": rmse(y_tree, tree_s_pred) - rmse(y_tree, piece_best["pred"]),
        "xi_piecewise_gain_over_smooth": rmse(y_xi, xi_smooth) - rmse(y_xi, piece_best["xi"]),
        "xi_twofreq_gain_over_smooth": rmse(y_xi, xi_smooth) - rmse(y_xi, xi_twofreq),
    }
    decision_path = OUTDIR / f"hll_uv_action_loop_eymh_parent_etree_regime_fit_decision_{TAG}.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (PAPER / decision_path.name).write_text(json.dumps(decision, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(d, y_tree, "o-", color="#5b2a86", label="E_tree data")
    ax.plot(d, tree_s_pred, lw=2, color="#2a9d8f", label="smooth")
    ax.plot(d, tree_tf_pred, lw=2, color="#f3722c", label="two-frequency global")
    ax.plot(d, piece_best["pred"], lw=2, color="#577590", ls="--", label=f"piecewise (split={piece_best['split']:.1f})")
    ax.set_title("E_tree regime fits")
    ax.set_xlabel("D")
    ax.set_ylabel("E_tree")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(d, y_xi, "o-", color="k", label="xi_target")
    ax.plot(d, xi_smooth, lw=2, color="#2a9d8f", label="part smooth × tree smooth")
    ax.plot(d, xi_twofreq, lw=2, color="#f3722c", label="part smooth × tree twofreq")
    ax.plot(d, piece_best["xi"], lw=2, color="#577590", ls="--", label="part smooth × tree piecewise")
    ax.set_title("xi reconstruction")
    ax.set_xlabel("D")
    ax.set_ylabel("xi_target")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(d, np.abs(y_tree - tree_s_pred), lw=2, color="#2a9d8f", label="smooth residual")
    ax.plot(d, np.abs(y_tree - tree_tf_pred), lw=2, color="#f3722c", label="twofreq residual")
    ax.plot(d, np.abs(y_tree - piece_best["pred"]), lw=2, color="#577590", ls="--", label="piecewise residual")
    ax.set_title("Absolute E_tree residual by D")
    ax.set_xlabel("D")
    ax.set_ylabel("|residual|")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(d, np.abs(y_xi - xi_smooth), lw=2, color="#2a9d8f", label="xi smooth residual")
    ax.plot(d, np.abs(y_xi - xi_twofreq), lw=2, color="#f3722c", label="xi twofreq residual")
    ax.plot(d, np.abs(y_xi - piece_best["xi"]), lw=2, color="#577590", ls="--", label="xi piecewise residual")
    ax.set_title("Absolute xi residual by D")
    ax.set_xlabel("D")
    ax.set_ylabel("|residual|")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("EYMH E_tree advanced functionalization: global two-frequency vs piecewise regime fit", fontsize=15)
    fig_path = OUTDIR / f"hll_uv_action_loop_eymh_parent_etree_regime_fit_{TAG}.png"
    fig.savefig(fig_path, dpi=220)
    fig.savefig(PAPER / fig_path.name, dpi=220)
    plt.close(fig)

    print(fig_path)
    print(summary_path)
    print(prof_path)
    print(piece_path)
    print(decision_path)


if __name__ == "__main__":
    main()

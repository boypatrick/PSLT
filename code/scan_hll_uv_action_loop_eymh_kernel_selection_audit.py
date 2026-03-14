#!/usr/bin/env python3
"""
Kernel-selection audit for the parented EYMH projected response block.

This script asks a narrower question than the previous response-action and
log-det / Schur audits:

Why does the projected fluctuation operator select the specific

  S_resp = log det K_part + log G_schur

kernel, rather than a nearby deformed kernel with altered weights or an
explicit projected cross-coupling between the participation and tree blocks?

We probe the smallest nontrivial family

  K11 = exp(alpha * S_part) = K_part ** alpha
  K22 = exp(beta  * S_schur) = G_schur ** beta
  K12 = lambda * sqrt((K11 - 1) (K22 - 1))

and define the deformed projected kernel

  K_sel = [[K11, K12],
           [K12, K22]].

The canonical parented kernel is recovered at

  alpha = 1, beta = 1, lambda = 0.

If the fluctuation operator truly selects the log-det / Schur form, the
response-weight and prefactor residuals should be uniquely minimized at that
point, with positive local curvature in the deformation directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPERDIR = ROOT / "paper"


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def load_parented_dataframe(parented_map: Path, constant_map: Path) -> pd.DataFrame:
    parented = pd.read_csv(parented_map)
    const = pd.read_csv(constant_map)[["D", "eta", "mu_mumu_uv_rge"]].rename(
        columns={"mu_mumu_uv_rge": "mu_mumu_uv_rge_const"}
    )
    return parented.merge(const, on=["D", "eta"], how="left")


def kernel_family(
    s_part: np.ndarray,
    s_schur: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k11 = np.exp(float(alpha) * s_part)
    k22 = np.exp(float(beta) * s_schur)
    cross_scale = np.sqrt(np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0))
    k12 = float(lam) * cross_scale
    det = np.maximum(k11 * k22 - k12**2, 1e-30)
    s = np.log(det)
    w = np.exp(-0.5 * s)
    return k11, k22, k12, w


def objective_weight(
    s_part: np.ndarray,
    s_schur: np.ndarray,
    w_ref: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
) -> float:
    _, _, _, w = kernel_family(s_part, s_schur, alpha, beta, lam)
    resid = w - w_ref
    return float(np.mean(resid**2))


def finite_difference_hessian(
    s_part: np.ndarray,
    s_schur: np.ndarray,
    w_ref: np.ndarray,
    point: tuple[float, float, float],
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(point, dtype=float)
    n = len(x)
    h = np.zeros((n, n), dtype=float)
    g = np.zeros(n, dtype=float)

    f0 = objective_weight(s_part, s_schur, w_ref, *x)
    eye = np.eye(n)
    for i in range(n):
        fp = objective_weight(s_part, s_schur, w_ref, *(x + eps * eye[i]))
        fm = objective_weight(s_part, s_schur, w_ref, *(x - eps * eye[i]))
        g[i] = (fp - fm) / (2.0 * eps)
        h[i, i] = (fp - 2.0 * f0 + fm) / (eps**2)
        for j in range(i + 1, n):
            fpp = objective_weight(s_part, s_schur, w_ref, *(x + eps * eye[i] + eps * eye[j]))
            fpm = objective_weight(s_part, s_schur, w_ref, *(x + eps * eye[i] - eps * eye[j]))
            fmp = objective_weight(s_part, s_schur, w_ref, *(x - eps * eye[i] + eps * eye[j]))
            fmm = objective_weight(s_part, s_schur, w_ref, *(x - eps * eye[i] - eps * eye[j]))
            hij = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)
            h[i, j] = hij
            h[j, i] = hij
    return g, h


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit why the projected EYMH fluctuation operator selects the canonical log-det / Schur kernel."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--alpha-grid", default="0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2")
    ap.add_argument("--beta-grid", default="0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2")
    ap.add_argument("--lambda-grid", default="-0.6,-0.4,-0.2,-0.1,0.0,0.1,0.2,0.4,0.6")
    args = ap.parse_args()

    def parse_grid(spec: str) -> list[float]:
        return [float(x.strip()) for x in spec.split(",") if x.strip()]

    alphas = parse_grid(args.alpha_grid)
    betas = parse_grid(args.beta_grid)
    lambdas = parse_grid(args.lambda_grid)

    df = load_parented_dataframe(args.parented_map, args.constant_map)
    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    s_part = np.log(coeff_l1 / coeff_l2)
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    s_schur = np.log1p(chi_tree)

    w_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    hk_local = np.asarray(df["hk_loop_local_prefactor_diag"], dtype=float)
    shell_spread = np.maximum(np.asarray(df["shell_spread"], dtype=float), 1e-30)
    shell_access = np.sqrt(shell_spread / (1.0 + shell_spread))
    pref_ref = np.asarray(df["eymh_parented_prefactor_diag"], dtype=float)

    scan_rows: list[dict[str, float]] = []
    for alpha in alphas:
        for beta in betas:
            for lam in lambdas:
                k11, k22, k12, w = kernel_family(s_part, s_schur, alpha, beta, lam)
                pref = hk_local * shell_access * w
                w_resid = w - w_ref
                p_resid = pref - pref_ref
                scan_rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "lambda": lam,
                        "mean_abs_weight_residual": float(np.mean(np.abs(w_resid))),
                        "max_abs_weight_residual": float(np.max(np.abs(w_resid))),
                        "rmse_weight_residual": float(np.sqrt(np.mean(w_resid**2))),
                        "mean_abs_prefactor_residual": float(np.mean(np.abs(p_resid))),
                        "max_abs_prefactor_residual": float(np.max(np.abs(p_resid))),
                        "rmse_prefactor_residual": float(np.sqrt(np.mean(p_resid**2))),
                        "mean_k12_abs": float(np.mean(np.abs(k12))),
                        "mean_det_kernel": float(np.mean(np.maximum(k11 * k22 - k12**2, 1e-30))),
                    }
                )

    scan = pd.DataFrame(scan_rows)
    best = scan.sort_values(
        ["rmse_prefactor_residual", "rmse_weight_residual", "mean_abs_prefactor_residual"]
    ).iloc[0]
    canonical = scan[
        (np.isclose(scan["alpha"], 1.0)) & (np.isclose(scan["beta"], 1.0)) & (np.isclose(scan["lambda"], 0.0))
    ].iloc[0]
    runner_up = scan.sort_values(
        ["rmse_prefactor_residual", "rmse_weight_residual", "mean_abs_prefactor_residual"]
    ).iloc[1]

    g, h = finite_difference_hessian(s_part, s_schur, w_ref, (1.0, 1.0, 0.0))
    eigvals = np.linalg.eigvalsh(h)

    _, _, _, w_canonical = kernel_family(s_part, s_schur, 1.0, 1.0, 0.0)
    response_action = -2.0 * np.log(np.maximum(w_canonical, 1e-30))
    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    summary = {
        "n_points": int(len(df)),
        "best_alpha": float(best["alpha"]),
        "best_beta": float(best["beta"]),
        "best_lambda": float(best["lambda"]),
        "canonical_rmse_weight_residual": float(canonical["rmse_weight_residual"]),
        "canonical_rmse_prefactor_residual": float(canonical["rmse_prefactor_residual"]),
        "runner_up_alpha": float(runner_up["alpha"]),
        "runner_up_beta": float(runner_up["beta"]),
        "runner_up_lambda": float(runner_up["lambda"]),
        "runner_up_rmse_prefactor_residual": float(runner_up["rmse_prefactor_residual"]),
        "selection_gap_rmse_prefactor": float(runner_up["rmse_prefactor_residual"] - canonical["rmse_prefactor_residual"]),
        "selection_gap_rmse_weight": float(runner_up["rmse_weight_residual"] - canonical["rmse_weight_residual"]),
        "grad_alpha_at_canonical": float(g[0]),
        "grad_beta_at_canonical": float(g[1]),
        "grad_lambda_at_canonical": float(g[2]),
        "hessian_eig1": float(eigvals[0]),
        "hessian_eig2": float(eigvals[1]),
        "hessian_eig3": float(eigvals[2]),
        "corr(response_action, prefactor)": corr(response_action, pref_ref),
        "corr(response_action, abs_mu_diff_vs_const)": corr(response_action, mu_diff_vs_const),
        "canonical_participation_share_p50": q(s_part / np.maximum(response_action, 1e-30), 0.5),
        "canonical_schur_share_p50": q(s_schur / np.maximum(response_action, 1e-30), 0.5),
    }

    df["kernel_selection_response_action"] = response_action
    df["kernel_selection_response_weight"] = w_canonical
    df["kernel_selection_prefactor_recon"] = hk_local * shell_access * w_canonical
    df["kernel_selection_prefactor_residual"] = df["kernel_selection_prefactor_recon"] - pref_ref
    df["kernel_selection_participation_share"] = s_part / np.maximum(response_action, 1e-30)
    df["kernel_selection_schur_share"] = s_schur / np.maximum(response_action, 1e-30)
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_kernel_selection_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_kernel_selection_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_kernel_selection_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_kernel_selection_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_kernel_selection_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "tree_diag_susceptibility_parented",
            "coeff_participation_access_parented",
            "tree_diag_compressibility_parented",
            "kernel_selection_response_action",
            "kernel_selection_response_weight",
            "eymh_parented_prefactor_diag",
            "kernel_selection_prefactor_recon",
            "kernel_selection_prefactor_residual",
            "kernel_selection_participation_share",
            "kernel_selection_schur_share",
            "mu_mumu_uv_rge",
            "mu_mumu_uv_rge_const",
            "abs_mu_diff_vs_const",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "constant_map": str(args.constant_map),
                "alpha_grid": alphas,
                "beta_grid": betas,
                "lambda_grid": lambdas,
            },
            indent=2,
            sort_keys=True,
        )
    )

    alpha_beta = scan[scan["lambda"] == 0.0].pivot(index="alpha", columns="beta", values="rmse_prefactor_residual")
    lambda_slice = scan[(scan["alpha"] == 1.0) & (scan["beta"] == 1.0)].sort_values("lambda")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im = axes[0].imshow(
        alpha_beta.values,
        origin="lower",
        aspect="auto",
        extent=[alpha_beta.columns.min(), alpha_beta.columns.max(), alpha_beta.index.min(), alpha_beta.index.max()],
    )
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("alpha")
    axes[0].set_title("RMSE(prefactor residual), lambda=0")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    axes[1].plot(lambda_slice["lambda"], lambda_slice["rmse_prefactor_residual"], marker="o")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("RMSE(prefactor residual)")
    axes[1].set_title("Cross-coupling slice at alpha=beta=1")

    axes[2].scatter(df["kernel_selection_response_action"], pref_ref, s=12, alpha=0.7)
    axes[2].set_xlabel("canonical selected response action")
    axes[2].set_ylabel("eymh_parented_prefactor_diag")
    axes[2].set_title("Selected kernel vs prefactor")

    fig.suptitle("EYMH projected kernel-selection audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_scan, out_map, out_summary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

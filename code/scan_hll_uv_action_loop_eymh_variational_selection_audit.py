#!/usr/bin/env python3
"""
Projected effective-action / variational-selection audit for the canonical
EYMH log-det + Schur kernel.

This script packages the previous kernel-selection and stationarity audits into
a local projected effective-action statement. Around the canonical point

  (alpha, beta, lambda) = (1, 1, 0),

we define the variational selection gap

  DeltaGamma_sel(dalpha, dbeta, lambda)
    = 1/2 * [dalpha dbeta] H [dalpha dbeta]^T
      + C4 * lambda^4,

where H is the exact quadratic stationarity matrix in the (alpha,beta) block
and C4 is the exact quartic coefficient of the even lambda direction.

We then compare this local effective-action gap to the exact mismatch
functional

  J(alpha,beta,lambda) = < (A(alpha,beta,lambda) - A_ref)^2 >.

If the projected log-det / Schur kernel truly defines the canonical local
selection principle, DeltaGamma_sel should closely reproduce J in a local
neighborhood of the canonical point, with both minima pinned at
(1,1,0).
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


def kernel_weight(
    s_part: np.ndarray,
    s_schur: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
) -> np.ndarray:
    k11 = np.exp(float(alpha) * s_part)
    k22 = np.exp(float(beta) * s_schur)
    k12 = float(lam) * np.sqrt(np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0))
    det = np.maximum(k11 * k22 - k12**2, 1e-30)
    return np.exp(-0.5 * np.log(det))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit the local projected effective-action selection gap of the canonical EYMH log-det / Schur kernel."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--alpha-grid", default="0.9,0.95,1.0,1.05,1.1")
    ap.add_argument("--beta-grid", default="0.9,0.95,1.0,1.05,1.1")
    ap.add_argument("--lambda-grid", default="-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2")
    args = ap.parse_args()

    def parse_grid(spec: str) -> list[float]:
        return [float(x.strip()) for x in spec.split(",") if x.strip()]

    alphas = parse_grid(args.alpha_grid)
    betas = parse_grid(args.beta_grid)
    lambdas = parse_grid(args.lambda_grid)

    df = pd.read_csv(args.parented_map)
    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    s_part = np.log(coeff_l1 / coeff_l2)
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    s_schur = np.log1p(chi_tree)
    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )

    score_alpha = -0.5 * s_part
    score_beta = -0.5 * s_schur
    h_aa = 2.0 * np.mean((a_ref * score_alpha) ** 2)
    h_bb = 2.0 * np.mean((a_ref * score_beta) ** 2)
    h_ab = 2.0 * np.mean((a_ref * score_alpha) * (a_ref * score_beta))
    H = np.array([[h_aa, h_ab], [h_ab, h_bb]], dtype=float)

    k_part = coeff_l1 / coeff_l2
    g_schur = 1.0 + chi_tree
    xi_cross = np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0) / np.maximum(k_part * g_schur, 1e-30)
    C4 = float(np.mean(0.25 * (a_ref**2) * (xi_cross**2)))

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        for beta in betas:
            for lam in lambdas:
                w = kernel_weight(s_part, s_schur, alpha, beta, lam)
                j_exact = float(np.mean((w - a_ref) ** 2))
                dtheta = np.array([alpha - 1.0, beta - 1.0], dtype=float)
                gamma_var = float(0.5 * dtheta @ H @ dtheta + C4 * (lam**4))
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "lambda": lam,
                        "j_exact": j_exact,
                        "gamma_var": gamma_var,
                        "abs_gap": abs(j_exact - gamma_var),
                        "rel_gap": abs(j_exact - gamma_var) / max(j_exact, 1e-30),
                    }
                )

    scan = pd.DataFrame(rows)
    best_exact = scan.sort_values(["j_exact", "abs_gap"]).iloc[0]
    best_var = scan.sort_values(["gamma_var", "abs_gap"]).iloc[0]
    lam_slice = scan[(scan["alpha"] == 1.0) & (scan["beta"] == 1.0)].sort_values("lambda")
    ab_slice = scan[scan["lambda"] == 0.0].copy()

    summary = {
        "n_scan_points": int(len(scan)),
        "best_exact_alpha": float(best_exact["alpha"]),
        "best_exact_beta": float(best_exact["beta"]),
        "best_exact_lambda": float(best_exact["lambda"]),
        "best_var_alpha": float(best_var["alpha"]),
        "best_var_beta": float(best_var["beta"]),
        "best_var_lambda": float(best_var["lambda"]),
        "canonical_exact_objective": float(
            scan[(scan["alpha"] == 1.0) & (scan["beta"] == 1.0) & (scan["lambda"] == 0.0)]["j_exact"].iloc[0]
        ),
        "canonical_variational_objective": float(
            scan[(scan["alpha"] == 1.0) & (scan["beta"] == 1.0) & (scan["lambda"] == 0.0)]["gamma_var"].iloc[0]
        ),
        "corr(j_exact, gamma_var)": corr(scan["j_exact"], scan["gamma_var"]),
        "max_abs_gap": float(scan["abs_gap"].max()),
        "p95_abs_gap": q(scan["abs_gap"], 0.95),
        "mean_abs_gap": float(scan["abs_gap"].mean()),
        "lambda_slice_max_abs_gap": float(lam_slice["abs_gap"].max()),
        "lambda_slice_p95_abs_gap": q(lam_slice["abs_gap"], 0.95),
        "ab_slice_max_abs_gap": float(ab_slice["abs_gap"].max()),
        "ab_slice_p95_abs_gap": q(ab_slice["abs_gap"], 0.95),
        "stationarity_h_eig1": float(np.linalg.eigvalsh(H)[0]),
        "stationarity_h_eig2": float(np.linalg.eigvalsh(H)[1]),
        "stationarity_c4": C4,
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_variational_selection_scan_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_variational_selection_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_variational_selection_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_variational_selection_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "alpha_grid": alphas,
                "beta_grid": betas,
                "lambda_grid": lambdas,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].scatter(scan["j_exact"], scan["gamma_var"], s=18, alpha=0.7)
    maxv = float(max(scan["j_exact"].max(), scan["gamma_var"].max(), 1e-12))
    axes[0].plot([0.0, maxv], [0.0, maxv], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("exact objective J")
    axes[0].set_ylabel("variational gap DeltaGamma_sel")
    axes[0].set_title("Exact vs variational")

    axes[1].plot(lam_slice["lambda"], lam_slice["j_exact"], marker="o", label="exact")
    axes[1].plot(lam_slice["lambda"], lam_slice["gamma_var"], marker="s", label="variational")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("selection gap")
    axes[1].set_title("lambda slice at alpha=beta=1")
    axes[1].legend(frameon=False)

    heat = ab_slice.pivot(index="alpha", columns="beta", values="abs_gap")
    im = axes[2].imshow(
        heat.values,
        origin="lower",
        aspect="auto",
        extent=[heat.columns.min(), heat.columns.max(), heat.index.min(), heat.index.max()],
    )
    axes[2].set_xlabel("beta")
    axes[2].set_ylabel("alpha")
    axes[2].set_title("|J - DeltaGamma_sel| at lambda=0")
    fig.colorbar(im, ax=axes[2], shrink=0.85)

    fig.suptitle("EYMH projected variational-selection audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_scan, out_summary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

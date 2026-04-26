#!/usr/bin/env python3
"""
Feshbach-Schur complement estimate for the projected EYMH parent kernel.

The existing parent-kernel audits show that the localized two-mode response
weight is

    A_proj = exp[-(log K11 + log G_schur) / 2].

If integrating out the complement Q produced a finite Feshbach correction, then

    H_eff = H_PP - H_PQ H_QQ^{-1} H_QP
          = H_PP^{1/2} (I - A_F) H_PP^{1/2},

and the exact low-mode response would differ by

    A_eff / A_proj = det(I - A_F)^(-1/2).

This audit does not claim a global spectral theorem for the full complement.
It measures the exported parented response against the projected logdet-Schur
response and converts the observed determinant-factor residual into the
equivalent Feshbach norm budget

    ||A_F||_eq = ell / (2 + ell),
    ell = |log det(I - A_F)|,

obtained by inverting the two-dimensional bound
|log det(I-A_F)| <= 2 ||A_F|| / (1 - ||A_F||).
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


def safe_log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(num, 1e-300) / np.maximum(den, 1e-300))


def equivalent_norm_budget(logdet_abs: np.ndarray) -> np.ndarray:
    # Inverts ell = 2 r / (1-r).  This is a certificate-size proxy, not
    # an independent measurement of the true complement operator norm.
    ell = np.asarray(logdet_abs, dtype=float)
    return ell / (2.0 + ell)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit the Feshbach-Schur complement budget of the projected EYMH parent kernel."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument(
        "--guard-norm",
        type=float,
        default=1e-12,
        help="Reviewer-facing complement-norm guard used to compare with the observed residual.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.parented_map)

    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-300)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-300)
    k11 = coeff_l1 / coeff_l2
    s_part = np.log(k11)

    k_bg = 1.0 + np.maximum(np.asarray(df["gap_cv"], dtype=float), 0.0)
    k_full = k_bg + np.maximum(np.asarray(df["c_tree_diag_cv"], dtype=float), 0.0)
    g_schur = k_full / np.maximum(k_bg, 1e-300)
    s_schur = np.log(g_schur)

    s_proj = s_part + s_schur
    a_proj = np.exp(-0.5 * s_proj)
    a_export = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )

    # If a_export = a_proj * det(I-A_F)^(-1/2), this is log det(I-A_F).
    logdet_factor_response = -2.0 * safe_log_ratio(a_export, a_proj)
    abs_logdet_factor_response = np.abs(logdet_factor_response)
    response_eq_norm = equivalent_norm_budget(abs_logdet_factor_response)

    shell_spread = np.maximum(np.asarray(df["shell_spread"], dtype=float), 1e-300)
    shell_access = np.sqrt(shell_spread / (1.0 + shell_spread))
    hk_local = np.asarray(df["hk_loop_local_prefactor_diag"], dtype=float)
    pref_proj = hk_local * shell_access * a_proj
    pref_export = np.asarray(df["eymh_parented_prefactor_diag"], dtype=float)

    logdet_factor_prefactor = -2.0 * safe_log_ratio(pref_export, pref_proj)
    abs_logdet_factor_prefactor = np.abs(logdet_factor_prefactor)
    pref_eq_norm = equivalent_norm_budget(abs_logdet_factor_prefactor)

    guard_norm = float(args.guard_norm)
    guard_logdet_budget = 2.0 * guard_norm / max(1.0 - guard_norm, 1e-300)

    out = df[["D", "eta"]].copy()
    out["feshbach_k11"] = k11
    out["feshbach_g_schur"] = g_schur
    out["feshbach_projected_action"] = s_proj
    out["feshbach_projected_response"] = a_proj
    out["feshbach_exported_response"] = a_export
    out["feshbach_response_logdet_factor"] = logdet_factor_response
    out["feshbach_abs_response_logdet_factor"] = abs_logdet_factor_response
    out["feshbach_response_equiv_norm"] = response_eq_norm
    out["feshbach_projected_prefactor"] = pref_proj
    out["feshbach_exported_prefactor"] = pref_export
    out["feshbach_prefactor_logdet_factor"] = logdet_factor_prefactor
    out["feshbach_abs_prefactor_logdet_factor"] = abs_logdet_factor_prefactor
    out["feshbach_prefactor_equiv_norm"] = pref_eq_norm
    out["feshbach_response_rel_drift"] = np.abs(a_export / np.maximum(a_proj, 1e-300) - 1.0)
    out["feshbach_prefactor_rel_drift"] = np.abs(pref_export / np.maximum(pref_proj, 1e-300) - 1.0)
    out["feshbach_positive_psd_sign_violation_response"] = np.maximum(logdet_factor_response, 0.0)
    out["feshbach_positive_psd_sign_violation_prefactor"] = np.maximum(logdet_factor_prefactor, 0.0)

    summary = {
        "n_points": int(len(out)),
        "guard_norm": guard_norm,
        "guard_logdet_budget": guard_logdet_budget,
        "response_max_abs_logdet_factor": float(np.max(abs_logdet_factor_response)),
        "response_p95_abs_logdet_factor": q(abs_logdet_factor_response, 0.95),
        "response_max_equiv_norm": float(np.max(response_eq_norm)),
        "response_p95_equiv_norm": q(response_eq_norm, 0.95),
        "response_max_relative_drift": float(np.max(out["feshbach_response_rel_drift"])),
        "response_p95_relative_drift": q(out["feshbach_response_rel_drift"], 0.95),
        "response_max_positive_psd_sign_violation": float(
            np.max(out["feshbach_positive_psd_sign_violation_response"])
        ),
        "prefactor_max_abs_logdet_factor": float(np.max(abs_logdet_factor_prefactor)),
        "prefactor_p95_abs_logdet_factor": q(abs_logdet_factor_prefactor, 0.95),
        "prefactor_max_equiv_norm": float(np.max(pref_eq_norm)),
        "prefactor_p95_equiv_norm": q(pref_eq_norm, 0.95),
        "prefactor_max_relative_drift": float(np.max(out["feshbach_prefactor_rel_drift"])),
        "prefactor_p95_relative_drift": q(out["feshbach_prefactor_rel_drift"], 0.95),
        "prefactor_max_positive_psd_sign_violation": float(
            np.max(out["feshbach_positive_psd_sign_violation_prefactor"])
        ),
        "response_guard_pass": bool(np.max(abs_logdet_factor_response) <= guard_logdet_budget),
        "prefactor_guard_pass": bool(np.max(abs_logdet_factor_prefactor) <= guard_logdet_budget),
        "response_guard_margin": float(
            guard_logdet_budget / max(float(np.max(abs_logdet_factor_response)), 1e-300)
        ),
        "prefactor_guard_margin": float(
            guard_logdet_budget / max(float(np.max(abs_logdet_factor_prefactor)), 1e-300)
        ),
        "corr(projected_action, response_abs_logdet_factor)": corr(s_proj, abs_logdet_factor_response),
        "corr(projected_action, prefactor_abs_logdet_factor)": corr(s_proj, abs_logdet_factor_prefactor),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_feshbach_complement_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_feshbach_complement_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_feshbach_complement_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_feshbach_complement_run_meta_{args.tag}.json"

    out.to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "guard_norm": guard_norm,
                "meaning": "observed projected/exported residual converted to an equivalent Feshbach norm budget",
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    axes[0].scatter(out["feshbach_projected_action"], abs_logdet_factor_response, s=12, alpha=0.7)
    axes[0].axhline(guard_logdet_budget, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("projected action")
    axes[0].set_ylabel("|log det complement|")
    axes[0].set_title("Response-level loss")

    axes[1].scatter(out["D"], response_eq_norm, s=12, alpha=0.7, label="response")
    axes[1].scatter(out["D"], pref_eq_norm, s=12, alpha=0.7, label="prefactor")
    axes[1].axhline(guard_norm, linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("D")
    axes[1].set_ylabel("equivalent ||A_F||")
    axes[1].set_title("Equivalent Feshbach norm budget")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].scatter(
        out["feshbach_abs_response_logdet_factor"],
        out["feshbach_abs_prefactor_logdet_factor"],
        s=12,
        alpha=0.7,
    )
    maxv = float(
        max(
            out["feshbach_abs_response_logdet_factor"].max(),
            out["feshbach_abs_prefactor_logdet_factor"].max(),
            1e-18,
        )
    )
    axes[2].plot([0.0, maxv], [0.0, maxv], linestyle="--", color="black", linewidth=1)
    axes[2].set_xlabel("response |logdet factor|")
    axes[2].set_ylabel("prefactor |logdet factor|")
    axes[2].set_title("Kernel vs full-prefactor audit")

    fig.suptitle("EYMH projected parent-kernel Feshbach-Schur complement audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_map, out_summary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

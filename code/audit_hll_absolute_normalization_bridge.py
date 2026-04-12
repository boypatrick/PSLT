#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from scan_hll_signal_strengths import make_baseline_kinetics  # noqa: E402


@dataclass(frozen=True)
class BridgeContext:
    x_mu: np.ndarray
    r_e_over_mu: np.ndarray
    r_tau_over_mu: np.ndarray
    mu_target: np.ndarray
    br: np.ndarray
    br_sum: float
    width_scale: float


def fit_scalar_amplitude(
    ctx: BridgeContext,
    r_e_star: float,
    r_tau_star: float,
    phi: np.ndarray,
) -> tuple[float, float]:
    def objective(log_amp: float) -> float:
        amp = float(np.exp(log_amp))
        mu = bridge_mu(
            ctx=ctx,
            amp_abs=amp,
            r_e_star=r_e_star,
            r_tau_star=r_tau_star,
            phi=phi,
        )
        return float(np.mean((mu - ctx.mu_target) ** 2))

    bounds = (np.log(1e5), np.log(1e12))
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(np.exp(result.x)), float(result.fun)


def fit_full_bridge(
    ctx: BridgeContext,
    init_amp_abs: float,
    init_r_e_star: float,
    init_r_tau_star: float,
    phi: np.ndarray,
) -> tuple[float, float, float, float]:
    def objective(theta: np.ndarray) -> float:
        amp_abs = float(np.exp(theta[0]))
        r_e_star = float(np.exp(theta[1]))
        r_tau_star = float(np.exp(theta[2]))
        mu = bridge_mu(
            ctx=ctx,
            amp_abs=amp_abs,
            r_e_star=r_e_star,
            r_tau_star=r_tau_star,
            phi=phi,
        )
        return float(np.mean((mu - ctx.mu_target) ** 2))

    bounds = [
        (np.log(1e5), np.log(1e12)),
        (np.log(1e-5), np.log(1.0)),
        (np.log(1e-5), np.log(1.0)),
    ]
    init = np.log([init_amp_abs, init_r_e_star, init_r_tau_star])
    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    amp_abs, r_e_star, r_tau_star = np.exp(result.x)
    return float(amp_abs), float(r_e_star), float(r_tau_star), float(result.fun)


def fit_bridge_with_fixed_re(
    ctx: BridgeContext,
    fixed_r_e_star: float,
    init_amp_abs: float,
    init_r_tau_star: float,
    phi: np.ndarray,
) -> tuple[float, float, float]:
    def objective(theta: np.ndarray) -> float:
        amp_abs = float(np.exp(theta[0]))
        r_tau_star = float(np.exp(theta[1]))
        mu = bridge_mu(
            ctx=ctx,
            amp_abs=amp_abs,
            r_e_star=fixed_r_e_star,
            r_tau_star=r_tau_star,
            phi=phi,
        )
        return float(np.mean((mu - ctx.mu_target) ** 2))

    bounds = [
        (np.log(1e5), np.log(1e12)),
        (np.log(1e-5), np.log(1.0)),
    ]
    init = np.log([init_amp_abs, init_r_tau_star])
    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    amp_abs, r_tau_star = np.exp(result.x)
    return float(amp_abs), float(r_tau_star), float(result.fun)


def bridge_mu(
    ctx: BridgeContext,
    amp_abs: float,
    r_e_star: float,
    r_tau_star: float,
    phi: np.ndarray,
) -> np.ndarray:
    local_partial_mu = float(amp_abs) * np.asarray(phi, dtype=float) * ctx.x_mu
    flavor_mix = (
        ctx.br[1]
        + ctx.br[0] * (ctx.r_e_over_mu / max(float(r_e_star), 1e-30))
        + ctx.br[2] * (ctx.r_tau_over_mu / max(float(r_tau_star), 1e-30))
    )
    width_ratio = 1.0 + float(ctx.width_scale) * (local_partial_mu * flavor_mix - ctx.br_sum)
    return local_partial_mu / np.maximum(width_ratio, 1e-30)


def summarize_candidate(
    name: str,
    calibration: str,
    phi_label: str,
    amp_abs: float,
    r_e_star: float,
    r_tau_star: float,
    phi: np.ndarray,
    ctx: BridgeContext,
    mu_obs: float,
    sigma_obs: float,
) -> tuple[dict[str, float | str], np.ndarray]:
    mu = bridge_mu(
        ctx=ctx,
        amp_abs=amp_abs,
        r_e_star=r_e_star,
        r_tau_star=r_tau_star,
        phi=phi,
    )
    diff = mu - ctx.mu_target
    abs_diff = np.abs(diff)
    chi2 = ((mu - float(mu_obs)) / max(float(sigma_obs), 1e-30)) ** 2
    chi2_target = ((ctx.mu_target - float(mu_obs)) / max(float(sigma_obs), 1e-30)) ** 2
    accept = chi2 <= 4.0
    accept_target = chi2_target <= 4.0

    summary = {
        "candidate": name,
        "calibration": calibration,
        "phi_label": phi_label,
        "amp_abs": float(amp_abs),
        "r_e_star": float(r_e_star),
        "r_tau_star": float(r_tau_star),
        "phi_norm_p50": float(np.quantile(phi, 0.5)),
        "phi_norm_p90": float(np.quantile(phi, 0.9)),
        "mean_abs_delta_mu": float(np.mean(abs_diff)),
        "p95_abs_delta_mu": float(np.quantile(abs_diff, 0.95)),
        "max_abs_delta_mu": float(np.max(abs_diff)),
        "rmse_delta_mu": float(np.sqrt(np.mean(diff * diff))),
        "acceptance_mismatch_fraction": float(np.mean(accept != accept_target)),
        "f_chi2_le_4_bridge": float(np.mean(accept)),
        "f_chi2_le_4_target": float(np.mean(accept_target)),
        "best_chi2_bridge": float(np.min(chi2)),
        "best_chi2_target": float(np.min(chi2_target)),
    }
    return summary, mu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an anchor-free H->mumu absolute-normalization bridge on top of the "
            "canonical action_loop_eymh_parented UV->EFT map."
        )
    )
    parser.add_argument(
        "--map-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--invariant-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "reference_anchor_invariant_ratios.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_df = pd.read_csv(args.map_input)
    summary_df = pd.read_csv(args.summary_input)
    invariant_df = pd.read_csv(args.invariant_input)

    summary_row = summary_df.iloc[0]
    accept_row = invariant_df[invariant_df["scope"] == "chi2_accept"].iloc[0]

    d_values = np.sort(map_df["D"].unique())
    d_num = int(len(d_values))
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=str(summary_row["chain_mode"]),
        d_min=float(np.min(d_values)),
        d_max=float(np.max(d_values)),
        d_num=d_num,
        uv_blend=float(summary_row["uv_blend"]),
        uv_m2_power=float(summary_row["uv_m2_power"]),
        uv_match_kappa_diag=float(summary_row["uv_match_kappa_diag"]),
        uv_match_kappa_offdiag=float(summary_row["uv_match_kappa_offdiag"]),
        uv_match_mode=str(summary_row["uv_match_mode"]),
        uv_match_input_diag_scale=float(summary_row["uv_match_input_diag_scale"]),
        uv_match_input_offdiag_scale=float(summary_row["uv_match_input_offdiag_scale"]),
        uv_rge_mu_low=float(summary_row["uv_rge_mu_low"]),
        uv_rge_gamma_diag=float(summary_row["uv_rge_gamma_diag"]),
        uv_rge_gamma_offdiag=float(summary_row["uv_rge_gamma_offdiag"]),
        uv_rge_log_clip=float(summary_row["uv_rge_log_clip"]),
    )

    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    t_coh = 1.0
    n_max = 20
    c_ref_diag = np.diag(kin.hll_wilson_matrix_uv_rge(ref_d, ref_eta, t_coh, n_max))
    width_cfg = kin._hll_match_config()
    br = np.array([width_cfg.br_ee, width_cfg.br_mumu, width_cfg.br_tautau], dtype=float)

    x_mu = map_df["C_ir_mumu"].to_numpy(dtype=float) ** 2
    r_e_over_mu = (
        map_df["C_ir_ee"].to_numpy(dtype=float) ** 2 / np.maximum(x_mu, 1e-300)
    )
    r_tau_over_mu = (
        map_df["C_ir_tautau"].to_numpy(dtype=float) ** 2 / np.maximum(x_mu, 1e-300)
    )
    mu_target = map_df["mu_mumu_uv_rge"].to_numpy(dtype=float)
    ctx = BridgeContext(
        x_mu=x_mu,
        r_e_over_mu=r_e_over_mu,
        r_tau_over_mu=r_tau_over_mu,
        mu_target=mu_target,
        br=br,
        br_sum=float(np.sum(br)),
        width_scale=float(width_cfg.width_scale),
    )

    amp_abs_ref = 1.0 / max(float(c_ref_diag[1] ** 2), 1e-300)
    r_e_star_ref = float((c_ref_diag[0] / max(c_ref_diag[1], 1e-300)) ** 2)
    r_tau_star_ref = float((c_ref_diag[2] / max(c_ref_diag[1], 1e-300)) ** 2)
    r_e_star_med = float(accept_row["R_e_over_mu_p50"])
    r_tau_star_med = float(accept_row["R_tau_over_mu_p50"])

    phi_const = np.ones(len(map_df), dtype=float)
    phi_parented = map_df["eymh_parented_prefactor_diag"].to_numpy(dtype=float)
    phi_parented /= max(float(np.median(phi_parented)), 1e-300)
    phi_action = map_df["action_abs_diag"].to_numpy(dtype=float)
    phi_action /= max(float(np.median(phi_action)), 1e-300)

    rows: list[dict[str, float | str]] = []
    detail = map_df[["D", "eta", "C_ir_ee", "C_ir_mumu", "C_ir_tautau", "mu_mumu_uv_rge"]].copy()
    detail["x_mu_abs2"] = x_mu
    detail["R_e_over_mu"] = r_e_over_mu
    detail["R_tau_over_mu"] = r_tau_over_mu

    exact_summary, mu_exact = summarize_candidate(
        name="exact_reference_invariant_bridge",
        calibration="point_from_live_reference",
        phi_label="none",
        amp_abs=amp_abs_ref,
        r_e_star=r_e_star_ref,
        r_tau_star=r_tau_star_ref,
        phi=phi_const,
        ctx=ctx,
        mu_obs=float(summary_row["mu_obs"]),
        sigma_obs=float(summary_row["sigma_obs"]),
    )
    rows.append(exact_summary)
    detail["mu_exact_reference_invariant_bridge"] = mu_exact
    detail["abs_delta_exact_reference_invariant_bridge"] = np.abs(mu_exact - mu_target)

    amp_fit, r_e_fit, r_tau_fit, _ = fit_full_bridge(
        ctx=ctx,
        init_amp_abs=amp_abs_ref,
        init_r_e_star=r_e_star_med,
        init_r_tau_star=r_tau_star_med,
        phi=phi_const,
    )
    fit_summary, mu_fit = summarize_candidate(
        name="three_parameter_global_invariant_fit",
        calibration="global_mse_fit",
        phi_label="none",
        amp_abs=amp_fit,
        r_e_star=r_e_fit,
        r_tau_star=r_tau_fit,
        phi=phi_const,
        ctx=ctx,
        mu_obs=float(summary_row["mu_obs"]),
        sigma_obs=float(summary_row["sigma_obs"]),
    )
    rows.append(fit_summary)
    detail["mu_three_parameter_global_invariant_fit"] = mu_fit
    detail["abs_delta_three_parameter_global_invariant_fit"] = np.abs(mu_fit - mu_target)

    amp_fit_fixed_re, r_tau_fit_fixed_re, _ = fit_bridge_with_fixed_re(
        ctx=ctx,
        fixed_r_e_star=r_e_star_med,
        init_amp_abs=amp_abs_ref,
        init_r_tau_star=r_tau_star_ref,
        phi=phi_const,
    )
    fixed_re_summary, mu_fixed_re = summarize_candidate(
        name="two_parameter_bridge_fixed_re_median",
        calibration="global_mse_fit_with_fixed_re_median",
        phi_label="none",
        amp_abs=amp_fit_fixed_re,
        r_e_star=r_e_star_med,
        r_tau_star=r_tau_fit_fixed_re,
        phi=phi_const,
        ctx=ctx,
        mu_obs=float(summary_row["mu_obs"]),
        sigma_obs=float(summary_row["sigma_obs"]),
    )
    rows.append(fixed_re_summary)
    detail["mu_two_parameter_bridge_fixed_re_median"] = mu_fixed_re
    detail["abs_delta_two_parameter_bridge_fixed_re_median"] = np.abs(mu_fixed_re - mu_target)

    for candidate_name, phi_label, phi in [
        ("one_scalar_accepted_median_invariant_bridge", "none", phi_const),
        ("one_scalar_accepted_median_invariant_bridge_parented_phi", "eymh_parented_prefactor_diag_p50_norm", phi_parented),
        ("one_scalar_accepted_median_invariant_bridge_actionabs_phi", "action_abs_diag_p50_norm", phi_action),
    ]:
        amp_fit_scalar, _ = fit_scalar_amplitude(
            ctx=ctx,
            r_e_star=r_e_star_med,
            r_tau_star=r_tau_star_med,
            phi=phi,
        )
        summary, mu = summarize_candidate(
            name=candidate_name,
            calibration="scalar_mse_fit",
            phi_label=phi_label,
            amp_abs=amp_fit_scalar,
            r_e_star=r_e_star_med,
            r_tau_star=r_tau_star_med,
            phi=phi,
            ctx=ctx,
            mu_obs=float(summary_row["mu_obs"]),
            sigma_obs=float(summary_row["sigma_obs"]),
        )
        rows.append(summary)
        detail[f"mu_{candidate_name}"] = mu
        detail[f"abs_delta_{candidate_name}"] = np.abs(mu - mu_target)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_out = outdir / "hll_absolute_normalization_bridge_summary.csv"
    detail_out = outdir / "hll_absolute_normalization_bridge_detail.csv"
    constants_out = outdir / "hll_absolute_normalization_bridge_constants.csv"
    meta_out = outdir / "hll_absolute_normalization_bridge_run_meta.json"

    pd.DataFrame(rows).to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)

    constants = pd.DataFrame(
        [
            {
                "source": "live_reference_point",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "amp_abs": amp_abs_ref,
                "r_e_star": r_e_star_ref,
                "r_tau_star": r_tau_star_ref,
                "mu_obs": float(summary_row["mu_obs"]),
                "sigma_obs": float(summary_row["sigma_obs"]),
            },
            {
                "source": "accepted_invariant_medians",
                "ref_D": np.nan,
                "ref_eta": np.nan,
                "amp_abs": np.nan,
                "r_e_star": r_e_star_med,
                "r_tau_star": r_tau_star_med,
                "mu_obs": float(summary_row["mu_obs"]),
                "sigma_obs": float(summary_row["sigma_obs"]),
            },
        ]
    )
    constants.to_csv(constants_out, index=False)

    meta = {
        "map_input": str(args.map_input),
        "summary_input": str(args.summary_input),
        "invariant_input": str(args.invariant_input),
        "ref_D": ref_d,
        "ref_eta": ref_eta,
        "chain_mode": str(summary_row["chain_mode"]),
        "uv_match_mode": str(summary_row["uv_match_mode"]),
        "uv_rge_mu_low": float(summary_row["uv_rge_mu_low"]),
        "n_points": int(len(map_df)),
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()

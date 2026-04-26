#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from audit_hll_absolute_normalization_bridge import BridgeContext, bridge_mu  # noqa: E402
from scan_hll_signal_strengths import make_baseline_kinetics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-policy H->mumu map-impact gate for replacing the release "
            "y_2^{raw}(D_*) scalar by the direct inserted-history value."
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
        "--amp-constants-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_absolute_amp_source_constants.csv"
        ),
    )
    parser.add_argument(
        "--y2-constants-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_y2raw_refpoint_stability_constants.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    parser.add_argument("--copy-paper", action="store_true")
    return parser.parse_args()


def _read_y2_constants(path: str | Path) -> tuple[float, float]:
    constants = pd.read_csv(path)
    release = constants[constants["candidate"] == "current_release_reference"]
    direct = constants[constants["candidate"] == "canonical_history_direct_reference"]
    if release.empty or direct.empty:
        raise ValueError("y2 constants must contain release and canonical-history rows.")
    return float(release.iloc[0]["y2_raw_ref"]), float(direct.iloc[0]["y2_raw_ref"])


def _make_context(map_df: pd.DataFrame, summary_row: pd.Series) -> BridgeContext:
    d_values = np.sort(map_df["D"].unique())
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=str(summary_row["chain_mode"]),
        d_min=float(np.min(d_values)),
        d_max=float(np.max(d_values)),
        d_num=int(len(d_values)),
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
    width_cfg = kin._hll_match_config()
    br = np.array([width_cfg.br_ee, width_cfg.br_mumu, width_cfg.br_tautau], dtype=float)

    x_mu = map_df["C_ir_mumu"].to_numpy(dtype=float) ** 2
    r_e_over_mu = map_df["C_ir_ee"].to_numpy(dtype=float) ** 2 / np.maximum(x_mu, 1e-300)
    r_tau_over_mu = map_df["C_ir_tautau"].to_numpy(dtype=float) ** 2 / np.maximum(x_mu, 1e-300)
    return BridgeContext(
        x_mu=x_mu,
        r_e_over_mu=r_e_over_mu,
        r_tau_over_mu=r_tau_over_mu,
        mu_target=map_df["mu_mumu_uv_rge"].to_numpy(dtype=float),
        br=br,
        br_sum=float(np.sum(br)),
        width_scale=float(width_cfg.width_scale),
    )


def _case_summary(
    case: str,
    y2_ref: float,
    amp_abs: float,
    mu: np.ndarray,
    chi2: np.ndarray,
    accept: np.ndarray,
    map_df: pd.DataFrame,
    amp_release: float,
    y2_release: float,
) -> dict[str, float | int | str | bool]:
    best_idx = int(np.argmin(chi2))
    best = map_df.iloc[best_idx]
    return {
        "row_type": "case",
        "case": case,
        "y2_raw_ref": float(y2_ref),
        "amp_abs": float(amp_abs),
        "amp_over_release": float(amp_abs / max(float(amp_release), 1e-300)),
        "y2_over_release": float(y2_ref / max(float(y2_release), 1e-300)),
        "n_points": int(len(map_df)),
        "f_hmumu_chi2_le_4": float(np.mean(accept)),
        "best_chi2": float(chi2[best_idx]),
        "best_mu_mumu": float(mu[best_idx]),
        "best_D": float(best["D"]),
        "best_eta": float(best["eta"]),
        "strict_adoption_safe": bool(case == "release_interpolated_y2"),
    }


def _delta_summary(
    detail: pd.DataFrame,
    y2_release: float,
    y2_direct: float,
    amp_release: float,
    amp_direct: float,
) -> dict[str, float | int | str | bool]:
    abs_dmu = np.abs(detail["delta_mu_mumu_direct_minus_release"].to_numpy(dtype=float))
    accept_release = detail["accept_release"].to_numpy(dtype=bool)
    accept_direct = detail["accept_direct_y2"].to_numpy(dtype=bool)
    flip = accept_direct != accept_release
    gain = (~accept_release) & accept_direct
    loss = accept_release & (~accept_direct)
    strict_safe = bool(np.mean(flip) == 0.0)
    return {
        "row_type": "delta",
        "case": "direct_inserted_history_y2_minus_release",
        "y2_raw_release": float(y2_release),
        "y2_raw_direct": float(y2_direct),
        "rel_y2_direct_vs_release": float(y2_direct / max(y2_release, 1e-300) - 1.0),
        "amp_abs_release": float(amp_release),
        "amp_abs_direct": float(amp_direct),
        "rel_amp_direct_vs_release": float(amp_direct / max(amp_release, 1e-300) - 1.0),
        "n_points": int(len(detail)),
        "mean_abs_delta_mu_mumu": float(np.mean(abs_dmu)),
        "p95_abs_delta_mu_mumu": float(np.percentile(abs_dmu, 95.0)),
        "max_abs_delta_mu_mumu": float(np.max(abs_dmu)),
        "acceptance_mismatch": float(np.mean(flip)),
        "acceptance_gain_fraction": float(np.mean(gain)),
        "acceptance_loss_fraction": float(np.mean(loss)),
        "n_acceptance_flips": int(np.sum(flip)),
        "n_acceptance_gains": int(np.sum(gain)),
        "n_acceptance_losses": int(np.sum(loss)),
        "f_hmumu_chi2_le_4_release": float(np.mean(accept_release)),
        "f_hmumu_chi2_le_4_direct": float(np.mean(accept_direct)),
        "delta_f_hmumu_chi2_le_4": float(np.mean(accept_direct) - np.mean(accept_release)),
        "strict_adoption_safe": strict_safe,
        "adoption_verdict": "adoption_safe" if strict_safe else "diagnostic_only_acceptance_topology_changes",
    }


def main() -> None:
    args = parse_args()
    map_df = pd.read_csv(args.map_input).sort_values(["D", "eta"]).reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    amp_constants = pd.read_csv(args.amp_constants_input).iloc[0]
    y2_release, y2_direct = _read_y2_constants(args.y2_constants_input)

    ctx = _make_context(map_df, summary_row)
    amp_release = float(amp_constants["A_star_ir"])
    amp_direct = float(amp_release * (y2_release / max(y2_direct, 1e-300)) ** 2)
    r_e_star = float(amp_constants["R_e_over_mu_star_ref"])
    r_tau_star = float(amp_constants["R_tau_over_mu_star_ref"])
    phi = np.ones(len(map_df), dtype=float)

    mu_release = bridge_mu(
        ctx=ctx,
        amp_abs=amp_release,
        r_e_star=r_e_star,
        r_tau_star=r_tau_star,
        phi=phi,
    )
    mu_direct = bridge_mu(
        ctx=ctx,
        amp_abs=amp_direct,
        r_e_star=r_e_star,
        r_tau_star=r_tau_star,
        phi=phi,
    )

    mu_obs = float(summary_row["mu_obs"])
    sigma_obs = float(summary_row["sigma_obs"])
    chi2_release = ((mu_release - mu_obs) / max(sigma_obs, 1e-300)) ** 2
    chi2_direct = ((mu_direct - mu_obs) / max(sigma_obs, 1e-300)) ** 2
    accept_release = chi2_release <= 4.0
    accept_direct = chi2_direct <= 4.0

    detail = map_df[["D", "eta", "mu_mumu_uv_rge", "chi2_uv_rge"]].copy()
    detail["mu_mumu_release_bridge"] = mu_release
    detail["mu_mumu_direct_y2"] = mu_direct
    detail["delta_mu_mumu_direct_minus_release"] = mu_direct - mu_release
    detail["abs_delta_mu_mumu_direct_minus_release"] = np.abs(mu_direct - mu_release)
    detail["chi2_release_bridge"] = chi2_release
    detail["chi2_direct_y2"] = chi2_direct
    detail["accept_release"] = accept_release
    detail["accept_direct_y2"] = accept_direct
    detail["acceptance_flip"] = accept_direct != accept_release
    detail["acceptance_gain"] = (~accept_release) & accept_direct
    detail["acceptance_loss"] = accept_release & (~accept_direct)

    rows: list[dict[str, float | int | str | bool]] = [
        _case_summary(
            case="release_interpolated_y2",
            y2_ref=y2_release,
            amp_abs=amp_release,
            mu=mu_release,
            chi2=chi2_release,
            accept=accept_release,
            map_df=map_df,
            amp_release=amp_release,
            y2_release=y2_release,
        ),
        _case_summary(
            case="direct_inserted_history_y2",
            y2_ref=y2_direct,
            amp_abs=amp_direct,
            mu=mu_direct,
            chi2=chi2_direct,
            accept=accept_direct,
            map_df=map_df,
            amp_release=amp_release,
            y2_release=y2_release,
        ),
        _delta_summary(
            detail=detail,
            y2_release=y2_release,
            y2_direct=y2_direct,
            amp_release=amp_release,
            amp_direct=amp_direct,
        ),
    ]

    slice_summary = (
        detail.groupby("D", as_index=False)
        .agg(
            n_points=("eta", "size"),
            f_accept_release=("accept_release", "mean"),
            f_accept_direct=("accept_direct_y2", "mean"),
            acceptance_flip_fraction=("acceptance_flip", "mean"),
            acceptance_gain_fraction=("acceptance_gain", "mean"),
            acceptance_loss_fraction=("acceptance_loss", "mean"),
            mean_abs_delta_mu_mumu=("abs_delta_mu_mumu_direct_minus_release", "mean"),
            p95_abs_delta_mu_mumu=("abs_delta_mu_mumu_direct_minus_release", lambda x: float(np.percentile(np.asarray(x, dtype=float), 95.0))),
            max_abs_delta_mu_mumu=("abs_delta_mu_mumu_direct_minus_release", "max"),
        )
        .sort_values(["acceptance_flip_fraction", "max_abs_delta_mu_mumu"], ascending=[False, False])
    )

    constants = pd.DataFrame(
        [
            {
                "quantity": "y2_raw_release",
                "value": float(y2_release),
                "description": "release interpolated y_2^{raw}(D_*) scalar",
            },
            {
                "quantity": "y2_raw_direct_inserted_history",
                "value": float(y2_direct),
                "description": "direct D_* solve inserted into the full canonical branch history",
            },
            {
                "quantity": "A_star_release",
                "value": float(amp_release),
                "description": "live IR bridge amplitude reconstructed from the current map",
            },
            {
                "quantity": "A_star_direct_inserted_history_y2",
                "value": float(amp_direct),
                "description": "one-policy replacement A_* = A_release (y_release/y_direct)^2",
            },
            {
                "quantity": "R_e_over_mu_star_fixed",
                "value": float(r_e_star),
                "description": "held fixed during the gate",
            },
            {
                "quantity": "R_tau_over_mu_star_fixed",
                "value": float(r_tau_star),
                "description": "held fixed during the gate",
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_y2raw_adoption_impact_gate_summary.csv"
    detail_out = outdir / "hll_y2raw_adoption_impact_gate_detail.csv"
    slice_out = outdir / "hll_y2raw_adoption_impact_gate_slice_summary.csv"
    constants_out = outdir / "hll_y2raw_adoption_impact_gate_constants.csv"
    meta_out = outdir / "hll_y2raw_adoption_impact_gate_run_meta.json"

    pd.DataFrame(rows).to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    slice_summary.to_csv(slice_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "map_input": str(args.map_input),
                "summary_input": str(args.summary_input),
                "amp_constants_input": str(args.amp_constants_input),
                "y2_constants_input": str(args.y2_constants_input),
                "policy": "replace only y_2^{raw}(D_*) in A_*; hold ratios, map rows, kinetic/mass, dressing, kernel, and family fixed",
                "mu_obs": mu_obs,
                "sigma_obs": sigma_obs,
                "n_points": int(len(map_df)),
            },
            indent=2,
        )
    )

    if bool(args.copy_paper):
        paper_dir = ROOT / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        for src in (summary_out, detail_out, slice_out, constants_out, meta_out):
            dst = paper_dir / src.name
            dst.write_bytes(src.read_bytes())
            print(f"[copied] {dst}")

    print(summary_out)
    print(detail_out)
    print(slice_out)
    print(constants_out)
    print(meta_out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

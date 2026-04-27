#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from audit_hll_absolute_normalization_bridge import bridge_mu  # noqa: E402
from audit_hll_y2raw_adoption_impact_gate import (  # noqa: E402
    _make_context,
    _read_y2_constants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Projector-continuation certificate for the U10 y_2^{raw}(D_*) "
            "reference scalar.  The audit keeps the EYMH/H->mumu bridge fixed "
            "and varies only the scalar y_2^{raw} between the release "
            "interpolation and the direct inserted-history parent point."
        )
    )
    parser.add_argument(
        "--map-input",
        default=str(ROOT / "paper" / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv"),
    )
    parser.add_argument(
        "--summary-input",
        default=str(ROOT / "paper" / "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"),
    )
    parser.add_argument(
        "--amp-constants-input",
        default=str(ROOT / "output" / "hll_absolute_normalization" / "hll_absolute_amp_source_constants.csv"),
    )
    parser.add_argument(
        "--y2-constants-input",
        default=str(ROOT / "output" / "hll_absolute_normalization" / "hll_y2raw_refpoint_stability_constants.csv"),
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "hll_absolute_normalization"),
    )
    parser.add_argument("--copy-paper", action="store_true")
    return parser.parse_args()


def bridge_mu_from_amp(
    *,
    ctx,
    amp_abs: float,
    r_e_star: float,
    r_tau_star: float,
) -> np.ndarray:
    return bridge_mu(
        ctx=ctx,
        amp_abs=float(amp_abs),
        r_e_star=float(r_e_star),
        r_tau_star=float(r_tau_star),
        phi=np.ones(len(ctx.x_mu), dtype=float),
    )


def critical_amp_for_mu(
    *,
    x_mu: np.ndarray,
    flavor_mix: np.ndarray,
    width_scale: float,
    br_sum: float,
    mu_boundary: float,
) -> np.ndarray:
    """Solve mu = L / (a + b L), L=A*x_mu, for A at a fixed boundary."""
    a = 1.0 - float(width_scale) * float(br_sum)
    b = float(width_scale) * np.asarray(flavor_mix, dtype=float)
    denom = np.asarray(x_mu, dtype=float) * (1.0 - float(mu_boundary) * b)
    out = np.full_like(denom, np.nan, dtype=float)
    good = denom > 0.0
    out[good] = float(mu_boundary) * a / denom[good]
    return out


def main() -> None:
    args = parse_args()
    map_df = pd.read_csv(args.map_input).sort_values(["D", "eta"]).reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    amp_constants = pd.read_csv(args.amp_constants_input).iloc[0]
    y2_release, y2_direct = _read_y2_constants(args.y2_constants_input)

    if not y2_direct > y2_release > 0:
        raise ValueError("This audit expects y2_direct > y2_release > 0.")

    ctx = _make_context(map_df, summary_row)
    amp_release = float(amp_constants["A_star_ir"])
    amp_direct = float(amp_release * (y2_release / y2_direct) ** 2)
    r_e_star = float(amp_constants["R_e_over_mu_star_ref"])
    r_tau_star = float(amp_constants["R_tau_over_mu_star_ref"])

    mu_obs = float(summary_row["mu_obs"])
    sigma_obs = float(summary_row["sigma_obs"])
    lower = mu_obs - 2.0 * sigma_obs
    upper = mu_obs + 2.0 * sigma_obs
    log_y_total = float(np.log(y2_direct / y2_release))
    log_amp_total = float(np.log(amp_direct / amp_release))
    direct_s = 1.0

    flavor_mix = (
        ctx.br[1]
        + ctx.br[0] * (ctx.r_e_over_mu / max(r_e_star, 1e-300))
        + ctx.br[2] * (ctx.r_tau_over_mu / max(r_tau_star, 1e-300))
    )
    a_den = 1.0 - float(ctx.width_scale) * float(ctx.br_sum)
    b_den = float(ctx.width_scale) * flavor_mix

    mu_release = bridge_mu_from_amp(
        ctx=ctx,
        amp_abs=amp_release,
        r_e_star=r_e_star,
        r_tau_star=r_tau_star,
    )
    mu_direct = bridge_mu_from_amp(
        ctx=ctx,
        amp_abs=amp_direct,
        r_e_star=r_e_star,
        r_tau_star=r_tau_star,
    )
    chi2_release = ((mu_release - mu_obs) / max(sigma_obs, 1e-300)) ** 2
    chi2_direct = ((mu_direct - mu_obs) / max(sigma_obs, 1e-300)) ** 2
    accept_release = (mu_release >= lower) & (mu_release <= upper)
    accept_direct = (mu_direct >= lower) & (mu_direct <= upper)

    amp_cross_upper = critical_amp_for_mu(
        x_mu=ctx.x_mu,
        flavor_mix=flavor_mix,
        width_scale=float(ctx.width_scale),
        br_sum=float(ctx.br_sum),
        mu_boundary=upper,
    )
    amp_cross_lower = critical_amp_for_mu(
        x_mu=ctx.x_mu,
        flavor_mix=flavor_mix,
        width_scale=float(ctx.width_scale),
        br_sum=float(ctx.br_sum),
        mu_boundary=lower,
    )

    crossing_type: list[str] = []
    amp_cross = np.full(len(map_df), np.nan, dtype=float)
    s_cross = np.full(len(map_df), np.nan, dtype=float)
    mu_cross_boundary = np.full(len(map_df), np.nan, dtype=float)
    for i in range(len(map_df)):
        kind = "none"
        amp = np.nan
        boundary = np.nan
        if (mu_release[i] > upper) and (mu_direct[i] <= upper):
            kind = "upper_gain"
            amp = amp_cross_upper[i]
            boundary = upper
        elif (mu_release[i] >= lower) and (mu_direct[i] < lower):
            kind = "lower_loss"
            amp = amp_cross_lower[i]
            boundary = lower
        crossing_type.append(kind)
        amp_cross[i] = amp
        mu_cross_boundary[i] = boundary
        if np.isfinite(amp) and amp > 0:
            s_cross[i] = np.log(amp / amp_release) / log_amp_total

    y_cross = y2_release * np.exp(s_cross * log_y_total)
    amp_ratio_cross = amp_cross / amp_release
    validation_mu_cross = np.full(len(map_df), np.nan, dtype=float)
    for i, amp in enumerate(amp_cross):
        if np.isfinite(amp):
            validation_mu_cross[i] = bridge_mu_from_amp(
                ctx=ctx,
                amp_abs=float(amp),
                r_e_star=r_e_star,
                r_tau_star=r_tau_star,
            )[i]

    detail = map_df[["D", "eta"]].copy()
    detail["mu_release"] = mu_release
    detail["mu_direct"] = mu_direct
    detail["chi2_release"] = chi2_release
    detail["chi2_direct"] = chi2_direct
    detail["accept_release"] = accept_release
    detail["accept_direct"] = accept_direct
    detail["acceptance_flip"] = accept_release != accept_direct
    detail["crossing_type"] = crossing_type
    detail["s_cross"] = s_cross
    detail["log_y_cross"] = s_cross * log_y_total
    detail["y2_cross"] = y_cross
    detail["y2_cross_over_release"] = y_cross / y2_release
    detail["amp_cross"] = amp_cross
    detail["amp_cross_over_release"] = amp_ratio_cross
    detail["direct_s_minus_s_cross"] = direct_s - s_cross
    detail["mu_boundary"] = mu_cross_boundary
    detail["mu_cross_formula_validation"] = validation_mu_cross
    detail["abs_mu_cross_validation_error"] = np.abs(validation_mu_cross - mu_cross_boundary)
    detail["a_den"] = a_den
    detail["b_den"] = b_den

    crossing = detail[detail["crossing_type"] != "none"].copy()
    if crossing.empty:
        s_safe = 1.0
        verdict = "adoption_safe_no_projector_continuation_crossing"
        max_validation_error = 0.0
    else:
        s_safe = float(crossing["s_cross"].min())
        verdict = "obstructed_direct_projector_continuation_crosses_acceptance_boundary"
        max_validation_error = float(crossing["abs_mu_cross_validation_error"].max())

    log_y_safe = s_safe * log_y_total
    y2_safe = float(y2_release * np.exp(log_y_safe))
    amp_safe = float(amp_release * np.exp(s_safe * log_amp_total))

    summary_rows = [
        {
            "section": "projector_continuation_parameters",
            "n_points": int(len(map_df)),
            "mu_obs": mu_obs,
            "sigma_obs": sigma_obs,
            "accept_lower": lower,
            "accept_upper": upper,
            "y2_release": y2_release,
            "y2_direct": y2_direct,
            "rel_y2_direct_vs_release": float(y2_direct / y2_release - 1.0),
            "amp_release": amp_release,
            "amp_direct": amp_direct,
            "amp_direct_over_release": float(amp_direct / amp_release),
            "log_y_direct_over_release": log_y_total,
            "log_amp_direct_over_release": log_amp_total,
            "width_scale": float(ctx.width_scale),
            "br_sum": float(ctx.br_sum),
            "denominator_a_min": float(a_den),
            "denominator_b_min": float(np.min(b_den)),
            "denominator_b_max": float(np.max(b_den)),
        },
        {
            "section": "topology_crossing_certificate",
            "n_accept_release": int(np.sum(accept_release)),
            "n_accept_direct": int(np.sum(accept_direct)),
            "n_acceptance_flips": int(np.sum(accept_release != accept_direct)),
            "n_upper_gain_crossings": int(np.sum(detail["crossing_type"] == "upper_gain")),
            "n_lower_loss_crossings": int(np.sum(detail["crossing_type"] == "lower_loss")),
            "first_cross_s": s_safe,
            "first_cross_log_y": log_y_safe,
            "first_cross_y2": y2_safe,
            "first_cross_y2_over_release": float(y2_safe / y2_release),
            "first_cross_amp": amp_safe,
            "first_cross_amp_over_release": float(amp_safe / amp_release),
            "direct_s_over_first_cross_s": float(direct_s / s_safe) if s_safe > 0 else np.inf,
            "direct_log_y_over_first_cross_log_y": (
                float(log_y_total / log_y_safe) if abs(log_y_safe) > 0 else np.inf
            ),
            "max_abs_mu_cross_validation_error": max_validation_error,
            "verdict": verdict,
        },
    ]

    detail_with_flags = detail.assign(
        is_crossing=detail["crossing_type"] != "none",
        abs_delta_mu=np.abs(detail["mu_direct"] - detail["mu_release"]),
    )
    by_d = detail_with_flags.groupby("D", as_index=False).agg(
        n_points=("eta", "size"),
        n_accept_release=("accept_release", "sum"),
        n_accept_direct=("accept_direct", "sum"),
        n_acceptance_flips=("acceptance_flip", "sum"),
        n_crossings=("is_crossing", "sum"),
        min_s_cross=("s_cross", "min"),
        max_s_cross=("s_cross", "max"),
        mean_abs_delta_mu=("abs_delta_mu", "mean"),
        max_abs_delta_mu=("abs_delta_mu", "max"),
    )
    delta_by_d = detail_with_flags.groupby("D")["abs_delta_mu"]
    by_d["mean_abs_delta_mu"] = by_d["D"].map(delta_by_d.mean())
    by_d["max_abs_delta_mu"] = by_d["D"].map(delta_by_d.max())

    constants = pd.DataFrame(
        [
            {
                "quantity": "first_cross_projector_fraction_s",
                "value": s_safe,
                "description": "largest log-linear y2 continuation fraction before the first acceptance-boundary crossing",
            },
            {
                "quantity": "max_no_flip_y2_raw",
                "value": y2_safe,
                "description": "supremal y2 value at the first D21xE21 topology crossing; strict no-flip requires a smaller value",
            },
            {
                "quantity": "max_no_flip_y2_over_release",
                "value": float(y2_safe / y2_release),
                "description": "relative y2 movement allowed before the first topology crossing",
            },
            {
                "quantity": "min_no_flip_amp_over_release",
                "value": float(amp_safe / amp_release),
                "description": "relative A* movement at the same crossing",
            },
            {
                "quantity": "direct_y2_over_release",
                "value": float(y2_direct / y2_release),
                "description": "full inserted-history direct y2 movement",
            },
            {
                "quantity": "direct_amp_over_release",
                "value": float(amp_direct / amp_release),
                "description": "full inserted-history direct A* movement",
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_y2raw_projector_continuation_summary.csv"
    detail_out = outdir / "hll_y2raw_projector_continuation_detail.csv"
    slice_out = outdir / "hll_y2raw_projector_continuation_slice_summary.csv"
    constants_out = outdir / "hll_y2raw_projector_continuation_constants.csv"
    meta_out = outdir / "hll_y2raw_projector_continuation_run_meta.json"

    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    by_d.to_csv(slice_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "map_input": str(args.map_input),
                "summary_input": str(args.summary_input),
                "amp_constants_input": str(args.amp_constants_input),
                "y2_constants_input": str(args.y2_constants_input),
                "parameterization": "log-linear y2(s)=y_release*exp(s*log(y_direct/y_release)); A(s)=A_release*exp(-2*s*log(y_direct/y_release))",
                "verdict": verdict,
            },
            indent=2,
        )
    )

    if args.copy_paper:
        paper_dir = ROOT / "paper"
        for path in (summary_out, detail_out, slice_out, constants_out, meta_out):
            shutil.copy2(path, paper_dir / path.name)

    print(summary_out)
    print(detail_out)
    print(slice_out)
    print(constants_out)
    print(meta_out)
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()

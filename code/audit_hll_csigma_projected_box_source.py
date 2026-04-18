#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from audit_hll_csigma_source import d2_rho, d2_z  # noqa: E402
from audit_hll_csigma_projected_overlap import (  # noqa: E402
    evaluate_projected_curve,
    load_or_solve_mode_block,
    microcanonical_weights,
    parse_d_subset,
)
from extract_chi_localized_2d import Level, PhysicalParams  # noqa: E402
from extract_y_eff_2d_three_channel import OverlapConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the one-center EYMH c_sigma source already fixes the "
            "projected-Yukawa box normalization, and how much the analytic two-center "
            "mirror correction improves it."
        )
    )
    parser.add_argument(
        "--overlap-detail-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
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
        "--projected-summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_csigma_projected_overlap_summary.csv"
        ),
    )
    parser.add_argument(
        "--eymh-core-summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_csigma_eymh_core_source_summary.csv"
        ),
    )
    parser.add_argument(
        "--eymh-core-detail-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_csigma_eymh_core_source_detail.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/mode_cache",
    )
    parser.add_argument("--Ds", default="6,12,18")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=30000)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--h2", type=float, default=1.0e-3)
    return parser.parse_args()


def curve_value(
    *,
    c_value: float,
    kappa_iso: float,
    D: float,
    rho_max: float,
    z_margin: float,
    rr: np.ndarray,
    zz: np.ndarray,
    wpsi: np.ndarray,
    weights_mc: np.ndarray,
    chunk: int,
) -> float:
    out = evaluate_projected_curve(
        c_grid=np.array([c_value], dtype=float),
        kappa_iso=kappa_iso,
        D=D,
        rho_max=rho_max,
        z_margin=z_margin,
        rr=rr,
        zz=zz,
        wpsi=wpsi,
        weights_mc=weights_mc,
        chunk=chunk,
    )
    return float(out[0])


def main() -> None:
    args = parse_args()
    d_subset = parse_d_subset(args.Ds)
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    if d_subset:
        detail = detail[detail["D"].isin(d_subset)].copy().sort_values("D").reset_index(drop=True)
    if detail.empty:
        raise ValueError("No overlap-detail rows left after applying --Ds.")

    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])

    projected_summary = pd.read_csv(args.projected_summary_input)
    c_proj = float(
        projected_summary.loc[
            projected_summary["section"] == "projected_overlap_selector",
            "c_sigma_projected_global",
        ].iloc[0]
    )
    c_amp = float(
        projected_summary.loc[
            projected_summary["section"] == "projected_overlap_selector",
            "c_sigma_amp",
        ].iloc[0]
    )
    c_ref = float(
        projected_summary.loc[
            projected_summary["section"] == "projected_overlap_selector",
            "c_sigma_ref",
        ].iloc[0]
    )

    eymh_summary = pd.read_csv(args.eymh_core_summary_input)
    c_self = float(
        eymh_summary.loc[
            eymh_summary["section"] == "self_core_source",
            "c_sigma_self",
        ].iloc[0]
    )
    eymh_detail = pd.read_csv(args.eymh_core_detail_input).copy().sort_values("D").reset_index(drop=True)
    c_analytic_map = {
        float(row["D"]): float(row["c_sigma_analytic"])
        for _, row in eymh_detail.iterrows()
    }

    p = PhysicalParams()
    cache_dir = Path(args.cache_dir)

    rows: list[dict[str, float]] = []
    sq_self = 0.0
    sq_analytic = 0.0
    sq_proj = 0.0
    sq_amp = 0.0
    sq_ref = 0.0

    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        n_eigs = int(row["n_eigs"])
        sigma_exact = 0.5 * (float(row["sigma_l"]) + float(row["sigma_r"]))
        level = Level(name=str(row["level"]), dr=dr, dz=dz)
        cfg = OverlapConfig(
            sigma_l=float(row["sigma_l"]),
            sigma_r=float(row["sigma_r"]),
            frame_power=float(row["frame_power"]),
            n_track=3,
            n_eigs=n_eigs,
            window_k=int(row["window_k"]),
            window_gap_scale=float(row["window_gap_scale"]),
            window_sigma_mult=float(row["window_sigma_mult"]),
            window_floor=float(row["window_floor"]),
        )
        solved = load_or_solve_mode_block(
            row=row,
            p=p,
            level=level,
            cfg=cfg,
            tol=args.tol,
            maxiter=args.maxiter,
            sigma=None if args.sigma < 0 else float(args.sigma),
            cache_dir=cache_dir,
        )
        rho = np.asarray(solved["rho"], dtype=float)
        rr = np.asarray(solved["rr"], dtype=float)
        zz = np.asarray(solved["zz"], dtype=float)
        psi = np.asarray(solved["psi"], dtype=float)
        evals = np.asarray(solved["evals"], dtype=float)
        center_idx = int(row["track_idx_2"])
        weights_mc, sigma_mc, nwin = microcanonical_weights(evals, center_idx, cfg)
        w = 2.0 * math.pi * rho[:, None] * dr * dz
        wpsi = np.stack([(w * psi[:, :, k]).reshape(-1) for k in range(psi.shape[2])], axis=1)

        kappa_iso = (2.0 * d2_rho(D, p, args.h2) + d2_z(D, p, args.h2)) / 3.0

        y2_exact = float(row["y_eff_raw_2"])
        c_analytic = c_analytic_map[D]

        y2_self = curve_value(
            c_value=c_self,
            kappa_iso=kappa_iso,
            D=D,
            rho_max=rho_max,
            z_margin=z_margin,
            rr=rr,
            zz=zz,
            wpsi=wpsi,
            weights_mc=weights_mc,
            chunk=int(args.chunk),
        )
        y2_analytic = curve_value(
            c_value=c_analytic,
            kappa_iso=kappa_iso,
            D=D,
            rho_max=rho_max,
            z_margin=z_margin,
            rr=rr,
            zz=zz,
            wpsi=wpsi,
            weights_mc=weights_mc,
            chunk=int(args.chunk),
        )
        y2_proj = curve_value(
            c_value=c_proj,
            kappa_iso=kappa_iso,
            D=D,
            rho_max=rho_max,
            z_margin=z_margin,
            rr=rr,
            zz=zz,
            wpsi=wpsi,
            weights_mc=weights_mc,
            chunk=int(args.chunk),
        )
        y2_amp = curve_value(
            c_value=c_amp,
            kappa_iso=kappa_iso,
            D=D,
            rho_max=rho_max,
            z_margin=z_margin,
            rr=rr,
            zz=zz,
            wpsi=wpsi,
            weights_mc=weights_mc,
            chunk=int(args.chunk),
        )
        y2_ref = curve_value(
            c_value=c_ref,
            kappa_iso=kappa_iso,
            D=D,
            rho_max=rho_max,
            z_margin=z_margin,
            rr=rr,
            zz=zz,
            wpsi=wpsi,
            weights_mc=weights_mc,
            chunk=int(args.chunk),
        )

        sq_self += (math.log(max(y2_self, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2
        sq_analytic += (math.log(max(y2_analytic, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2
        sq_proj += (math.log(max(y2_proj, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2
        sq_amp += (math.log(max(y2_amp, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2
        sq_ref += (math.log(max(y2_ref, 1.0e-300)) - math.log(max(y2_exact, 1.0e-300))) ** 2

        rows.append(
            {
                "D": D,
                "sigma_exact": sigma_exact,
                "sigma_mc": sigma_mc,
                "nwin": nwin,
                "kappa_iso": kappa_iso,
                "c_sigma_self": c_self,
                "c_sigma_analytic": c_analytic,
                "c_sigma_projected": c_proj,
                "c_sigma_amp": c_amp,
                "c_sigma_ref": c_ref,
                "y2_exact": y2_exact,
                "y2_self": y2_self,
                "y2_analytic": y2_analytic,
                "y2_projected": y2_proj,
                "y2_amp": y2_amp,
                "y2_ref": y2_ref,
                "rel_y2_resid_self": float(abs(y2_self / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_y2_resid_analytic": float(abs(y2_analytic / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_y2_resid_projected": float(abs(y2_proj / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_y2_resid_amp": float(abs(y2_amp / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_y2_resid_ref": float(abs(y2_ref / max(y2_exact, 1.0e-300) - 1.0)),
                "rel_self_vs_projected_y2": float(abs(y2_self / max(y2_proj, 1.0e-300) - 1.0)),
                "rel_analytic_vs_projected_y2": float(abs(y2_analytic / max(y2_proj, 1.0e-300) - 1.0)),
            }
        )

    out_detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    nearest = out_detail.loc[(out_detail["D"] - ref_d).abs().idxmin()]

    nrows = len(out_detail)
    summary = pd.DataFrame(
        [
            {
                "section": "candidate_constants",
                "c_sigma_ref": c_ref,
                "c_sigma_amp": c_amp,
                "c_sigma_projected": c_proj,
                "c_sigma_self": c_self,
                "rel_drift_self_vs_projected": float(abs(c_self / c_proj - 1.0)),
                "rel_drift_self_vs_amp": float(abs(c_self / c_amp - 1.0)),
                "rel_drift_self_vs_ref": float(abs(c_self / c_ref - 1.0)),
            },
            {
                "section": "reference_candidate_metrics",
                "c": c_ref,
                "y2_log_rmse": float(np.sqrt(sq_ref / nrows)),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_ref"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_ref"].mean()),
            },
            {
                "section": "amplitude_candidate_metrics",
                "c": c_amp,
                "y2_log_rmse": float(np.sqrt(sq_amp / nrows)),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_amp"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_amp"].mean()),
            },
            {
                "section": "projected_global_candidate_metrics",
                "c": c_proj,
                "y2_log_rmse": float(np.sqrt(sq_proj / nrows)),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_projected"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_projected"].mean()),
            },
            {
                "section": "self_core_candidate_metrics",
                "c": c_self,
                "y2_log_rmse": float(np.sqrt(sq_self / nrows)),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_self"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_self"].mean()),
                "max_rel_self_vs_projected_y2": float(out_detail["rel_self_vs_projected_y2"].max()),
                "mean_rel_self_vs_projected_y2": float(out_detail["rel_self_vs_projected_y2"].mean()),
            },
            {
                "section": "analytic_two_center_candidate_metrics",
                "y2_log_rmse": float(np.sqrt(sq_analytic / nrows)),
                "max_rel_y2_resid": float(out_detail["rel_y2_resid_analytic"].max()),
                "mean_rel_y2_resid": float(out_detail["rel_y2_resid_analytic"].mean()),
                "max_rel_analytic_vs_projected_y2": float(out_detail["rel_analytic_vs_projected_y2"].max()),
                "mean_rel_analytic_vs_projected_y2": float(out_detail["rel_analytic_vs_projected_y2"].mean()),
            },
            {
                "section": "reference_point",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "nearest_detail_D": float(nearest["D"]),
                "y2_exact_ref": float(nearest["y2_exact"]),
                "y2_self_ref": float(nearest["y2_self"]),
                "y2_analytic_ref": float(nearest["y2_analytic"]),
                "y2_projected_ref": float(nearest["y2_projected"]),
                "rel_y2_resid_self_ref": float(nearest["rel_y2_resid_self"]),
                "rel_y2_resid_analytic_ref": float(nearest["rel_y2_resid_analytic"]),
                "rel_self_vs_projected_y2_ref": float(nearest["rel_self_vs_projected_y2"]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "projected_box_normalization_constant",
                "formula": "argmin_c sum_D [log y_2^pred(D;c)-log y_2^raw(D)]^2",
                "status": "benchmark selector on the projected-Yukawa box-normalized overlap functional",
                "value": c_proj,
            },
            {
                "candidate": "self_core_to_projected_box",
                "formula": (
                    "insert c_sigma^(self) into y_2^pred(D;c) with the finite-box Gaussian kernel "
                    "and compare directly to y_2^raw(D)"
                ),
                "status": "explicit one-center EYMH source statement",
                "value": c_self,
            },
            {
                "candidate": "analytic_two_center_to_projected_box",
                "formula": "insert c_sigma^(analytic)(D) pointwise into y_2^pred(D;c)",
                "status": "explicit mirror-corrected source statement",
                "value_ref": float(nearest["c_sigma_analytic"]),
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_csigma_projected_box_source_summary.csv"
    detail_out = outdir / "hll_csigma_projected_box_source_detail.csv"
    constants_out = outdir / "hll_csigma_projected_box_source_constants.csv"
    meta_out = outdir / "hll_csigma_projected_box_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    out_detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "projected_summary_input": str(args.projected_summary_input),
                "eymh_core_summary_input": str(args.eymh_core_summary_input),
                "eymh_core_detail_input": str(args.eymh_core_detail_input),
                "Ds_filter": sorted(d_subset),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "sigma": None if args.sigma < 0 else float(args.sigma),
                "tol": float(args.tol),
                "maxiter": int(args.maxiter),
                "chunk": int(args.chunk),
                "cache_dir": str(cache_dir),
            },
            indent=2,
        )
    )

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()

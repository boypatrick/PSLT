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

from extract_chi_localized_2d import PhysicalParams, u_potential  # noqa: E402
from extract_y_eff_2d_three_channel import chirality_profiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit parent-side source candidates for the bridge constant c_sigma "
            "using width-, amplitude-, and full-kernel matching."
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
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    parser.add_argument("--h2", type=float, default=1.0e-3)
    parser.add_argument("--cmin", type=float, default=18.0)
    parser.add_argument("--cmax", type=float, default=28.0)
    parser.add_argument("--nscan", type=int, default=4001)
    return parser.parse_args()


def u_scalar(D: float, rho: float, z: float, p: PhysicalParams) -> float:
    return float(u_potential(np.array([[rho]]), np.array([[z]]), D, p)[0, 0])


def d2_rho(D: float, p: PhysicalParams, h: float) -> float:
    z0 = 0.5 * D
    return (u_scalar(D, h, z0, p) - 2.0 * u_scalar(D, 0.0, z0, p) + u_scalar(D, -h, z0, p)) / (h * h)


def d2_z(D: float, p: PhysicalParams, h: float) -> float:
    z0 = 0.5 * D
    return (u_scalar(D, 0.0, z0 + h, p) - 2.0 * u_scalar(D, 0.0, z0, p) + u_scalar(D, 0.0, z0 - h, p)) / (h * h)


def finite_box_amplitude(D: float, sigma: float, rho_max: float, z_margin: float) -> float:
    i_rho = math.pi * sigma * sigma * (1.0 - math.exp(-(rho_max * rho_max) / (sigma * sigma)))
    i_z = 0.5 * math.sqrt(math.pi) * sigma * (
        math.erf((D + z_margin) / sigma) + math.erf(z_margin / sigma)
    )
    return math.exp(-(D * D) / (4.0 * sigma * sigma)) / max(i_rho * i_z, 1.0e-300)


def rel_sup_error(target: np.ndarray, approx: np.ndarray) -> float:
    return float(np.max(np.abs(target - approx)) / max(float(np.max(np.abs(target))), 1.0e-300))


def build_rows(detail: pd.DataFrame, p: PhysicalParams, h2: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        sigma_exact = 0.5 * (float(row["sigma_l"]) + float(row["sigma_r"]))

        kappa_rho = d2_rho(D, p, h2)
        kappa_z = d2_z(D, p, h2)
        kappa_iso = (2.0 * kappa_rho + kappa_z) / 3.0
        c_sigma_exact = sigma_exact * (kappa_iso ** 0.25)

        nr = int(round(rho_max / dr))
        nz = int(round(2.0 * z_max / dz))
        rho = (np.arange(nr) + 0.5) * dr
        z = -z_max + (np.arange(nz) + 0.5) * dz
        rr, zz = np.meshgrid(rho, z, indexing="ij")
        f_l, f_r = chirality_profiles(
            rr=rr,
            zz=zz,
            rho=rho,
            dr=dr,
            dz=dz,
            d_val=D,
            sigma_l=sigma_exact,
            sigma_r=sigma_exact,
        )
        kernel = f_l * f_r
        base_exact = np.exp(-(rr * rr + zz * zz) / (sigma_exact * sigma_exact))
        amp_exact = float(np.sum(kernel * base_exact) / max(np.sum(base_exact * base_exact), 1.0e-300))

        rows.append(
            {
                "D": D,
                "dr": dr,
                "dz": dz,
                "rho_max": rho_max,
                "z_max": z_max,
                "z_margin": z_margin,
                "sigma_exact": sigma_exact,
                "kappa_iso": kappa_iso,
                "c_sigma_exact": c_sigma_exact,
                "amp_exact": amp_exact,
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate(c: float, rows: pd.DataFrame, detail: pd.DataFrame) -> dict[str, float]:
    sigma_c = c * np.power(rows["kappa_iso"].to_numpy(dtype=float), -0.25)
    sigma_exact = rows["sigma_exact"].to_numpy(dtype=float)
    amp_exact = rows["amp_exact"].to_numpy(dtype=float)
    Dvals = rows["D"].to_numpy(dtype=float)
    rho_max = rows["rho_max"].to_numpy(dtype=float)
    z_margin = rows["z_margin"].to_numpy(dtype=float)

    amp_box = np.array(
        [finite_box_amplitude(D, s, rmax, zmg) for D, s, rmax, zmg in zip(Dvals, sigma_c, rho_max, z_margin)],
        dtype=float,
    )
    width_log_rmse = float(np.sqrt(np.mean((np.log(sigma_c) - np.log(sigma_exact)) ** 2)))
    amp_log_rmse = float(np.sqrt(np.mean((np.log(amp_box) - np.log(amp_exact)) ** 2)))
    width_rel_max = float(np.max(np.abs(sigma_c / sigma_exact - 1.0)))
    amp_rel_max = float(np.max(np.abs(amp_box / amp_exact - 1.0)))

    kernel_rel_sup = []
    for (_, rrow), sigma in zip(detail.iterrows(), sigma_c):
        D = float(rrow["D"])
        dr = float(rrow["dr"])
        dz = float(rrow["dz"])
        rho_max = float(rrow["rho_max"])
        z_max = float(rrow["z_max"])
        z_margin = z_max - D / 2.0
        sigma_exact_i = 0.5 * (float(rrow["sigma_l"]) + float(rrow["sigma_r"]))
        nr = int(round(rho_max / dr))
        nz = int(round(2.0 * z_max / dz))
        rho = (np.arange(nr) + 0.5) * dr
        z = -z_max + (np.arange(nz) + 0.5) * dz
        rr, zz = np.meshgrid(rho, z, indexing="ij")
        f_l, f_r = chirality_profiles(
            rr=rr, zz=zz, rho=rho, dr=dr, dz=dz, d_val=D,
            sigma_l=sigma_exact_i, sigma_r=sigma_exact_i,
        )
        kernel = f_l * f_r
        approx = finite_box_amplitude(D, float(sigma), rho_max, z_margin) * np.exp(-(rr * rr + zz * zz) / (sigma * sigma))
        kernel_rel_sup.append(rel_sup_error(kernel, approx))
    kernel_rel_sup = np.array(kernel_rel_sup, dtype=float)

    return {
        "c": float(c),
        "width_log_rmse": width_log_rmse,
        "amp_log_rmse": amp_log_rmse,
        "max_sigma_relerr": width_rel_max,
        "max_amp_relerr": amp_rel_max,
        "max_kernel_rel_sup": float(np.max(kernel_rel_sup)),
        "mean_kernel_rel_sup": float(np.mean(kernel_rel_sup)),
    }


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    p = PhysicalParams()

    rows = build_rows(detail, p, args.h2)
    c_ref = float((rows.loc[(rows["D"] - ref_d).abs().idxmin(), "c_sigma_exact"]))
    c_mean = float(rows["c_sigma_exact"].mean())

    c_grid = np.linspace(args.cmin, args.cmax, int(args.nscan), dtype=float)
    eval_rows = [evaluate_candidate(float(c), rows, detail) for c in c_grid]
    eval_df = pd.DataFrame(eval_rows)

    i_w = int(eval_df["width_log_rmse"].idxmin())
    i_a = int(eval_df["amp_log_rmse"].idxmin())
    i_k = int(eval_df["max_kernel_rel_sup"].idxmin())
    best_w = eval_df.loc[i_w]
    best_a = eval_df.loc[i_a]
    best_k = eval_df.loc[i_k]
    ref_eval = evaluate_candidate(c_ref, rows, detail)
    mean_eval = evaluate_candidate(c_mean, rows, detail)

    summary = pd.DataFrame([
        {
            "section": "bridge_constant_candidates",
            "c_sigma_ref": c_ref,
            "c_sigma_mean": c_mean,
            "c_sigma_rel_span": float((rows["c_sigma_exact"].max() - rows["c_sigma_exact"].min()) / rows["c_sigma_exact"].mean()),
            "c_sigma_best_width": float(best_w["c"]),
            "c_sigma_best_amp": float(best_a["c"]),
            "c_sigma_best_kernel": float(best_k["c"]),
            "rel_drift_best_width_vs_ref": float(abs(best_w["c"] / c_ref - 1.0)),
            "rel_drift_best_amp_vs_ref": float(abs(best_a["c"] / c_ref - 1.0)),
            "rel_drift_best_kernel_vs_ref": float(abs(best_k["c"] / c_ref - 1.0)),
        },
        {
            "section": "reference_calibration_metrics",
            **ref_eval,
        },
        {
            "section": "global_mean_metrics",
            **mean_eval,
        },
        {
            "section": "best_width_fit_metrics",
            **{k: float(v) for k, v in best_w.to_dict().items()},
        },
        {
            "section": "best_amplitude_fit_metrics",
            **{k: float(v) for k, v in best_a.to_dict().items()},
        },
        {
            "section": "best_kernel_fit_metrics",
            **{k: float(v) for k, v in best_k.to_dict().items()},
        },
    ])

    nearest_idx = int((rows["D"] - ref_d).abs().idxmin())
    nearest = rows.loc[nearest_idx]
    constants = pd.DataFrame([
        {
            "candidate": "reference_curvature_bridge",
            "formula": "c_sigma = sigma_mu(D_*) * kappa_iso(D_*)^(1/4)",
            "status": "single-point calibration already used in current paper",
            "value": c_ref,
        },
        {
            "candidate": "global_width_match",
            "formula": "argmin_c sum_D [log sigma_exact(D) - log(c kappa_iso(D)^(-1/4))]^2",
            "status": "best global width-source fit",
            "value": float(best_w["c"]),
        },
        {
            "candidate": "global_box_amplitude_match",
            "formula": "argmin_c sum_D [log A_disc(D) - log A_box(D;c)]^2",
            "status": "best finite-box core-to-box matching constant",
            "value": float(best_a["c"]),
        },
        {
            "candidate": "global_kernel_match",
            "formula": "argmin_c max_D relsup(K_exact(D), K_box(D;c))",
            "status": "best full-kernel matching constant",
            "value": float(best_k["c"]),
        },
        {
            "candidate": "recommended_parent_side_reading",
            "formula": "c_sigma selected by finite-box core-to-box matching, then refined by kernel-level check",
            "status": "current best EYMH-side source candidate",
            "value": float(best_a["c"]),
        },
    ])

    detail_out = rows.copy()
    for key in ("c", "width_log_rmse", "amp_log_rmse", "max_sigma_relerr", "max_amp_relerr", "max_kernel_rel_sup", "mean_kernel_rel_sup"):
        detail_out[f"ref_{key}"] = ref_eval[key]
    detail_out["sigma_from_ref_c"] = c_ref * np.power(detail_out["kappa_iso"].to_numpy(dtype=float), -0.25)
    detail_out["sigma_from_best_amp_c"] = float(best_a["c"]) * np.power(detail_out["kappa_iso"].to_numpy(dtype=float), -0.25)
    detail_out["sigma_from_best_kernel_c"] = float(best_k["c"]) * np.power(detail_out["kappa_iso"].to_numpy(dtype=float), -0.25)
    detail_out["relerr_sigma_best_amp"] = np.abs(detail_out["sigma_from_best_amp_c"] / detail_out["sigma_exact"] - 1.0)
    detail_out["relerr_sigma_best_kernel"] = np.abs(detail_out["sigma_from_best_kernel_c"] / detail_out["sigma_exact"] - 1.0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_csigma_source_summary.csv"
    detail_path = outdir / "hll_csigma_source_detail.csv"
    constants_out = outdir / "hll_csigma_source_constants.csv"
    meta_out = outdir / "hll_csigma_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    detail_out.to_csv(detail_path, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(json.dumps({
        "overlap_detail_input": str(args.overlap_detail_input),
        "summary_input": str(args.summary_input),
        "ref_D": ref_d,
        "ref_eta": ref_eta,
        "nearest_detail_D": float(nearest["D"]),
        "cmin": args.cmin,
        "cmax": args.cmax,
        "nscan": int(args.nscan),
    }, indent=2))

    print(summary_out)
    print(detail_path)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()

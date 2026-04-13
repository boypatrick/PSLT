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

from extract_y_eff_2d_three_channel import OverlapConfig, flavor_sigma_scales  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit parent-side source candidates for y_2^{raw}(D_*), "
            "the overlap block controlling the canonical C_{mumu,*}^{tree} source."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy()
    summary_row = pd.read_csv(args.summary_input).iloc[0]

    cfg = OverlapConfig()
    sigma_scales = flavor_sigma_scales(cfg)

    detail["raw_vs_mu1_abs"] = (detail["y_eff_raw_1"] - detail["y_eff_flavor_mu_1"]).abs()
    detail["raw_vs_mu2_abs"] = (detail["y_eff_raw_2"] - detail["y_eff_flavor_mu_2"]).abs()
    detail["raw_vs_mu3_abs"] = (detail["y_eff_raw_3"] - detail["y_eff_flavor_mu_3"]).abs()
    detail["guv_mu1_sq_vs_raw_abs"] = (detail["g_uv_mu_1"] ** 2 - detail["y_eff_raw_1"]).abs()
    detail["guv_mu2_sq_vs_raw_abs"] = (detail["g_uv_mu_2"] ** 2 - detail["y_eff_raw_2"]).abs()
    detail["guv_mu3_sq_vs_raw_abs"] = (detail["g_uv_mu_3"] ** 2 - detail["y_eff_raw_3"]).abs()
    detail["center2_vs_raw2_abs"] = (detail["y_eff_center_2"] - detail["y_eff_raw_2"]).abs()
    detail["center2_vs_raw2_rel"] = detail["center2_vs_raw2_abs"] / detail["y_eff_raw_2"].abs().clip(lower=1e-30)
    detail["center2_over_raw2"] = detail["y_eff_center_2"] / detail["y_eff_raw_2"].clip(lower=1e-30)
    detail = detail.sort_values("D").reset_index(drop=True)

    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    if abs(float(summary_row["uv_blend"])) > 1e-15:
        raise ValueError("This audit expects the canonical uv_blend=0 diagonal-overlap baseline.")
    yraw_ref = float(np.interp(ref_d, detail["D"], detail["y_eff_raw_2"]))
    guv_ref = float(np.sqrt(max(yraw_ref, 0.0)))

    nearest_idx = (detail["D"] - ref_d).abs().idxmin()
    nearest_row = detail.loc[nearest_idx]

    summary = pd.DataFrame(
        [
            {
                "section": "exact_mu_flavor_identity",
                "sigma_scale_mu_expected": float(sigma_scales["mu"]),
                "sigma_scale_mu_min": float(detail["sigma_scale_mu"].min()),
                "sigma_scale_mu_max": float(detail["sigma_scale_mu"].max()),
                "max_abs_raw_vs_mu1": float(detail["raw_vs_mu1_abs"].max()),
                "max_abs_raw_vs_mu2": float(detail["raw_vs_mu2_abs"].max()),
                "max_abs_raw_vs_mu3": float(detail["raw_vs_mu3_abs"].max()),
                "max_abs_guv_mu1_sq_vs_raw": float(detail["guv_mu1_sq_vs_raw_abs"].max()),
                "max_abs_guv_mu2_sq_vs_raw": float(detail["guv_mu2_sq_vs_raw_abs"].max()),
                "max_abs_guv_mu3_sq_vs_raw": float(detail["guv_mu3_sq_vs_raw_abs"].max()),
            },
            {
                "section": "center_mode_candidate_check",
                "max_abs_center2_vs_raw2": float(detail["center2_vs_raw2_abs"].max()),
                "mean_abs_center2_vs_raw2": float(detail["center2_vs_raw2_abs"].mean()),
                "max_rel_center2_vs_raw2": float(detail["center2_vs_raw2_rel"].max()),
                "mean_rel_center2_vs_raw2": float(detail["center2_vs_raw2_rel"].mean()),
                "p95_rel_center2_vs_raw2": float(detail["center2_vs_raw2_rel"].quantile(0.95)),
                "nwin2_min": int(detail["nwin_2"].min()),
                "nwin2_max": int(detail["nwin_2"].max()),
                "sigma_mc2_min": float(detail["sigma_mc_2"].min()),
                "sigma_mc2_max": float(detail["sigma_mc_2"].max()),
            },
            {
                "section": "reference_overlap_target",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "y2_raw_ref_interp": yraw_ref,
                "g_uv_mu2_ref_interp": guv_ref,
                "abs_guv_mu2_sq_minus_y2_raw_ref": float(abs(guv_ref**2 - yraw_ref)),
                "nearest_detail_D": float(nearest_row["D"]),
                "nearest_detail_y2_raw": float(nearest_row["y_eff_raw_2"]),
                "nearest_detail_y2_center": float(nearest_row["y_eff_center_2"]),
                "nearest_detail_center2_vs_raw2_rel": float(nearest_row["center2_vs_raw2_rel"]),
                "nearest_detail_nwin2": int(nearest_row["nwin_2"]),
                "nearest_detail_sigma_mc2": float(nearest_row["sigma_mc_2"]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "primary_parent_source",
                "description": "microcanonical_windowed_mu_flavor_overlap",
                "formula": "y_2^raw(D)=sum_k w_{2k}(D) * |<u_k(D), K_mu(D)>|",
                "sigma_scale_mu": float(sigma_scales["mu"]),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "y2_raw_ref_interp": yraw_ref,
                "g_uv_mu2_ref_interp": guv_ref,
            },
            {
                "candidate": "equivalent_uv_coupling_source",
                "description": "g_uv_mu_2(D)^2",
                "formula": "g_{mu 2}^{UV}(D)^2 = y_2^raw(D)",
                "sigma_scale_mu": float(sigma_scales["mu"]),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "y2_raw_ref_interp": yraw_ref,
                "g_uv_mu2_ref_interp": guv_ref,
            },
            {
                "candidate": "rejected_coarse_candidate",
                "description": "center_mode_overlap_only",
                "formula": "y_{center,2}(D) = |<u_{k_2}(D), K_mu(D)>|",
                "nearest_detail_D": float(nearest_row["D"]),
                "nearest_detail_y2_raw": float(nearest_row["y_eff_raw_2"]),
                "nearest_detail_y2_center": float(nearest_row["y_eff_center_2"]),
                "nearest_detail_center2_vs_raw2_rel": float(nearest_row["center2_vs_raw2_rel"]),
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_y2raw_parent_source_summary.csv"
    detail_out = outdir / "hll_y2raw_parent_source_detail.csv"
    constants_out = outdir / "hll_y2raw_parent_source_constants.csv"
    meta_out = outdir / "hll_y2raw_parent_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "chain_mode": str(summary_row["chain_mode"]),
                "uv_blend": float(summary_row["uv_blend"]),
                "n_overlap_rows": int(len(detail)),
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

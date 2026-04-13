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

from extract_y_eff_2d_three_channel import chirality_profiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit minimal parent-side candidates for the canonical "
            "mu-flavor overlap kernel K_mu(D)."
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


def exact_midplane_amplitude(kernel: np.ndarray, base: np.ndarray) -> float:
    return float(np.sum(kernel * base) / max(np.sum(base * base), 1e-300))


def finite_box_amplitude(D: float, sigma: float, rho_max: float, z_margin: float) -> float:
    i_rho = math.pi * sigma * sigma * (1.0 - math.exp(-(rho_max * rho_max) / (sigma * sigma)))
    i_z = 0.5 * math.sqrt(math.pi) * sigma * (
        math.erf((D + z_margin) / sigma) + math.erf(z_margin / sigma)
    )
    return math.exp(-(D * D) / (4.0 * sigma * sigma)) / max(i_rho * i_z, 1e-300)


def infinite_volume_amplitude(D: float, sigma: float) -> float:
    return (math.pi ** (-1.5)) * (sigma ** -3.0) * math.exp(-(D * D) / (4.0 * sigma * sigma))


def relative_sup_error(target: np.ndarray, approx: np.ndarray) -> float:
    return float(np.max(np.abs(target - approx)) / max(float(np.max(np.abs(target))), 1e-300))


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])

    rows: list[dict[str, float]] = []
    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        sigma_l = float(row["sigma_l"])
        sigma_r = float(row["sigma_r"])
        frame_power = float(row["frame_power"])

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
            sigma_l=sigma_l,
            sigma_r=sigma_r,
        )
        kernel = f_l * f_r
        sigma_mu = 0.5 * (sigma_l + sigma_r)
        base = np.exp(-(rr * rr + zz * zz) / (sigma_mu * sigma_mu))

        amp_exact = exact_midplane_amplitude(kernel, base)
        amp_box = finite_box_amplitude(D, sigma_mu, rho_max, z_margin)
        amp_inf = infinite_volume_amplitude(D, sigma_mu)

        k_exact = amp_exact * base
        k_box = amp_box * base
        k_inf = amp_inf * base

        rows.append(
            {
                "D": D,
                "sigma_l": sigma_l,
                "sigma_r": sigma_r,
                "frame_power": frame_power,
                "rho_max": rho_max,
                "z_max": z_max,
                "z_margin": z_margin,
                "sigma_mu": sigma_mu,
                "amp_exact_disc": amp_exact,
                "amp_box_continuum": amp_box,
                "amp_infinite_volume": amp_inf,
                "rel_amp_box_vs_exact": float(abs(amp_box - amp_exact) / max(abs(amp_exact), 1e-300)),
                "rel_amp_inf_vs_exact": float(abs(amp_inf - amp_exact) / max(abs(amp_exact), 1e-300)),
                "rel_sup_exact_factorization": relative_sup_error(kernel, k_exact),
                "rel_sup_box_candidate": relative_sup_error(kernel, k_box),
                "rel_sup_infinite_candidate": relative_sup_error(kernel, k_inf),
                "mean_rel_box_candidate": float(np.mean(np.abs(kernel - k_box)) / max(float(np.mean(np.abs(kernel))), 1e-300)),
                "mean_rel_infinite_candidate": float(np.mean(np.abs(kernel - k_inf)) / max(float(np.mean(np.abs(kernel))), 1e-300)),
            }
        )

    out_detail = pd.DataFrame(rows)
    ref_idx = (out_detail["D"] - ref_d).abs().idxmin()
    nearest_row = out_detail.loc[ref_idx]

    sigma_mu_ref = float(out_detail["sigma_mu"].iloc[0])
    rho_max_ref = float(out_detail["rho_max"].iloc[0])
    dr_ref = float(detail["dr"].iloc[0])
    dz_ref = float(detail["dz"].iloc[0])
    z_margin_ref = float(out_detail["z_margin"].iloc[0])
    z_max_ref = ref_d / 2.0 + z_margin_ref
    nr_ref = int(round(rho_max_ref / dr_ref))
    nz_ref = int(round(2.0 * z_max_ref / dz_ref))
    rho_ref = (np.arange(nr_ref) + 0.5) * dr_ref
    z_ref = -z_max_ref + (np.arange(nz_ref) + 0.5) * dz_ref
    rr_ref, zz_ref = np.meshgrid(rho_ref, z_ref, indexing="ij")
    f_l_ref, f_r_ref = chirality_profiles(
        rr=rr_ref,
        zz=zz_ref,
        rho=rho_ref,
        dr=dr_ref,
        dz=dz_ref,
        d_val=ref_d,
        sigma_l=sigma_mu_ref,
        sigma_r=sigma_mu_ref,
    )
    kernel_ref = f_l_ref * f_r_ref
    base_ref = np.exp(-(rr_ref * rr_ref + zz_ref * zz_ref) / (sigma_mu_ref * sigma_mu_ref))
    amp_exact_ref = exact_midplane_amplitude(kernel_ref, base_ref)
    amp_box_ref = finite_box_amplitude(ref_d, sigma_mu_ref, rho_max_ref, z_margin_ref)
    amp_inf_ref = infinite_volume_amplitude(ref_d, sigma_mu_ref)
    rel_box_ref = relative_sup_error(kernel_ref, amp_box_ref * base_ref)
    rel_inf_ref = relative_sup_error(kernel_ref, amp_inf_ref * base_ref)

    summary = pd.DataFrame(
        [
            {
                "section": "canonical_kernel_metadata",
                "uv_blend": float(summary_row["uv_blend"]),
                "frame_power_min": float(out_detail["frame_power"].min()),
                "frame_power_max": float(out_detail["frame_power"].max()),
                "sigma_l_min": float(out_detail["sigma_l"].min()),
                "sigma_l_max": float(out_detail["sigma_l"].max()),
                "sigma_r_min": float(out_detail["sigma_r"].min()),
                "sigma_r_max": float(out_detail["sigma_r"].max()),
            },
            {
                "section": "exact_midplane_factorization",
                "max_rel_sup_exact_factorization": float(out_detail["rel_sup_exact_factorization"].max()),
                "mean_rel_sup_exact_factorization": float(out_detail["rel_sup_exact_factorization"].mean()),
            },
            {
                "section": "finite_box_continuum_candidate",
                "max_rel_sup_box_candidate": float(out_detail["rel_sup_box_candidate"].max()),
                "mean_rel_sup_box_candidate": float(out_detail["rel_sup_box_candidate"].mean()),
                "max_rel_amp_box_vs_exact": float(out_detail["rel_amp_box_vs_exact"].max()),
                "mean_rel_amp_box_vs_exact": float(out_detail["rel_amp_box_vs_exact"].mean()),
            },
            {
                "section": "infinite_volume_candidate",
                "max_rel_sup_infinite_candidate": float(out_detail["rel_sup_infinite_candidate"].max()),
                "mean_rel_sup_infinite_candidate": float(out_detail["rel_sup_infinite_candidate"].mean()),
                "max_rel_amp_inf_vs_exact": float(out_detail["rel_amp_inf_vs_exact"].max()),
                "mean_rel_amp_inf_vs_exact": float(out_detail["rel_amp_inf_vs_exact"].mean()),
            },
            {
                "section": "reference_candidate_constants",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "nearest_detail_D": float(nearest_row["D"]),
                "sigma_mu_ref": sigma_mu_ref,
                "amp_exact_disc_ref": amp_exact_ref,
                "amp_box_continuum_ref": amp_box_ref,
                "amp_infinite_volume_ref": amp_inf_ref,
                "rel_sup_box_candidate_ref": rel_box_ref,
                "rel_sup_infinite_candidate_ref": rel_inf_ref,
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "exact_discrete_midplane_bridge",
                "formula": "K_mu(rho,z;D)=A_mu^disc(D) * exp(-(rho^2+z^2)/sigma_mu^2)",
                "status": "exact_on_canonical_discrete_kernel",
                "sigma_mu_ref": sigma_mu_ref,
                "amp_ref": amp_exact_ref,
            },
            {
                "candidate": "finite_box_continuum_midplane_bridge",
                "formula": "A_mu^box(D)=exp(-D^2/(4 sigma_mu^2)) / [I_rho(sigma_mu,rho_max) I_z(sigma_mu,D,z_margin)]",
                "status": "best_parent_side_closed_form_candidate",
                "sigma_mu_ref": sigma_mu_ref,
                "amp_ref": amp_box_ref,
            },
            {
                "candidate": "infinite_volume_midplane_bridge",
                "formula": "A_mu^inf(D)=pi^(-3/2) sigma_mu^(-3) exp(-D^2/(4 sigma_mu^2))",
                "status": "too_coarse_for_current_box_normalized_extractor",
                "sigma_mu_ref": sigma_mu_ref,
                "amp_ref": amp_inf_ref,
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_kmu_parent_candidate_summary.csv"
    detail_out = outdir / "hll_kmu_parent_candidate_detail.csv"
    constants_out = outdir / "hll_kmu_parent_candidate_constants.csv"
    meta_out = outdir / "hll_kmu_parent_candidate_run_meta.json"

    summary.to_csv(summary_out, index=False)
    out_detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "n_overlap_rows": int(len(out_detail)),
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

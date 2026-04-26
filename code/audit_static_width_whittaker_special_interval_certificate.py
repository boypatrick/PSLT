#!/usr/bin/env python3
"""
Special-function interval certificate for the S24 parent derivative constants.

S24 promotes the final D=6, R_tail=400, R=60->80 tube by using one
parity-uniform derivative constant Lambda_pi on the whole interval.  This audit
keeps the same Whittaker-DtN backend and checks the constants against direct
special-function evaluations of

    d/dR G_pi(R,z),

where G_pi is the moving-coordinate implicit-flow quotient.  No new roots,
families, backends, or R-splits are introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_adapted_flow import parse_floats, parse_strings
from audit_static_width_whittaker_cauchy_slab_bound import HFlowEvaluator
from audit_static_width_whittaker_curved_flowbox import polynomial_centerline
from extract_static_width_exact_dtn import ExactDtnConfig
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_S22_SLABS = OUTDIR / "static_width_whittaker_slab_envelope_d6rc400_r60_r80_c640_slabs.csv"
DEFAULT_S24_INTERVALS = OUTDIR / "static_width_whittaker_interval_promotion_d6rc400_r60_r80_c640_summary.csv"
DEFAULT_TAG = "static_width_whittaker_special_interval_certificate_d6rc400_r60_r80_c640"


def parse_segments(raw: str) -> tuple[tuple[float, float], ...]:
    if raw.strip() == "":
        return ()
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        left, right = item.strip().split(":")
        out.append((float(left), float(right)))
    return tuple(out)


def z_offsets(radius: float, mode: str) -> tuple[complex, ...]:
    r = float(radius)
    if mode == "corners_center":
        return (0.0j, r + 1j * r, -r + 1j * r, r - 1j * r, -r - 1j * r)
    if mode == "octant_center":
        return (
            0.0j,
            r + 0.0j,
            -r + 0.0j,
            1j * r,
            -1j * r,
            r + 1j * r,
            -r + 1j * r,
            r - 1j * r,
            -r - 1j * r,
        )
    raise ValueError(f"Unknown --z-skeleton: {mode}")


def audit_parity(
    path: pd.DataFrame,
    slab_rows: pd.DataFrame,
    s24_row: pd.Series,
    center_mode: str,
    z_skeleton: str,
    dR_step: float,
    dE_step_fraction: float,
    cauchy_R_step_fraction: float,
    direct_guard_factor: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    path = path.sort_values("R").reset_index(drop=True)
    slabs = slab_rows.sort_values("slab_index").reset_index(drop=True)
    parity = str(slabs["parity"].iloc[0])
    D = float(slabs["D"].iloc[0])
    R_tail = float(slabs["R_tail"].iloc[0])
    ell = int(round(float(path.iloc[0].get("ell", cfg.ell))))
    n_half = int(round(float(path.iloc[0].get("n_half", cfg.n_half))))
    R0 = float(slabs["R0"].iloc[0])
    R1 = float(slabs["R1"].iloc[0])
    radius = float(slabs["tube_radius_min"].iloc[0])
    h_slab = float(slabs["delta_R"].iloc[0])
    dE_step = max(radius * float(dE_step_fraction), 1.0e-8)
    cauchy_R_step = h_slab * float(cauchy_R_step_fraction)
    s24_L = float(s24_row["max_interval_L_bound"])

    gamma, gamma_prime, anchors = polynomial_centerline(path=path, R0=R0, R1=R1, mode=center_mode)
    evaluator = HFlowEvaluator(
        D=D,
        R_tail=R_tail,
        parity=parity,
        n_half=n_half,
        ell=ell,
        gamma=gamma,
        gamma_prime=gamma_prime,
        cfg=cfg,
        p=p,
        dR_step=float(dR_step),
        dE_step=dE_step,
    )
    offsets = z_offsets(radius, z_skeleton)

    sample_rows: list[dict[str, float | str]] = []
    slab_rows_out: list[dict[str, float | str]] = []
    sampled_max_by_slab: list[float] = []

    for _, slab in slabs.iterrows():
        slab_index = int(round(float(slab["slab_index"])))
        R_left = float(slab["R_left"])
        R_right = float(slab["R_right"])
        R_mid = 0.5 * (R_left + R_right)
        slab_sample_max = 0.0
        den_min = np.inf
        numerator_max = 0.0
        cr_max = 0.0

        for station, R in (("left", R_left), ("mid", R_mid), ("right", R_right)):
            for point_index, z in enumerate(offsets):
                vals = evaluator.dH_dR_abs(R=R, z=complex(z), cauchy_R_step=cauchy_R_step)
                dH = float(vals["dH_dR_abs"])
                slab_sample_max = max(slab_sample_max, dH)
                den_min = min(den_min, float(vals["dF_dE_abs_min"]))
                numerator_max = max(numerator_max, float(vals["curved_numerator_abs_max"]))
                cr_max = max(cr_max, float(vals["dF_dE_CR_rel_mismatch_max"]))
                sample_rows.append(
                    {
                        "D": D,
                        "R_tail": R_tail,
                        "parity": parity,
                        "R0": R0,
                        "R1": R1,
                        "center_mode": center_mode,
                        "center_anchors": ";".join(f"{x:g}" for x in anchors),
                        "tube_radius_min": radius,
                        "slab_index": float(slab_index),
                        "station": station,
                        "R": float(R),
                        "point_index": float(point_index),
                        "z_real": float(complex(z).real),
                        "z_imag": float(complex(z).imag),
                        "dH_dR_abs": dH,
                        "s24_interval_L_bound": s24_L,
                        "dH_over_s24_L": dH / max(s24_L, 1.0e-300),
                        **vals,
                    }
                )

        sampled_max_by_slab.append(slab_sample_max)
        direct_L = float(direct_guard_factor) * slab_sample_max
        remainder = 0.5 * h_slab * direct_L
        reserve = 0.05 * float(slab["node_upper_beta"])
        beta_direct = float(slab["node_upper_beta"]) + remainder
        slab_rows_out.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "tube_radius_min": radius,
                "slab_index": float(slab_index),
                "R_left": R_left,
                "R_right": R_right,
                "delta_R": h_slab,
                "node_upper_beta": float(slab["node_upper_beta"]),
                "sampled_direct_dH_dR_abs_max": slab_sample_max,
                "direct_guard_factor": float(direct_guard_factor),
                "direct_L_bound": direct_L,
                "s24_interval_L_bound": s24_L,
                "direct_L_over_s24_L": direct_L / max(s24_L, 1.0e-300),
                "five_percent_reserve": reserve,
                "direct_remainder_bound": remainder,
                "direct_remainder_over_reserve": remainder / max(reserve, 1.0e-300),
                "beta_direct": beta_direct,
                "beta_s24_interval": float(slab["node_upper_beta"]) + 0.5 * h_slab * s24_L,
                "direct_contribution": h_slab * beta_direct,
                "s24_interval_contribution": h_slab * (float(slab["node_upper_beta"]) + 0.5 * h_slab * s24_L),
                "dF_dE_abs_min": float(den_min),
                "curved_numerator_abs_max": float(numerator_max),
                "dF_dE_CR_rel_mismatch_max": float(cr_max),
                "special_interval_pass": float(direct_L <= s24_L and beta_direct <= float(slab["inflated_beta"])),
            }
        )

    sample_df = pd.DataFrame(sample_rows)
    slab_df = pd.DataFrame(slab_rows_out)
    direct_ratio = float(slab_df["direct_contribution"].sum() / radius)
    s24_ratio = float(slab_df["s24_interval_contribution"].sum() / radius)
    n_pass = float(slab_df["special_interval_pass"].sum())
    summary = {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R0,
        "R1": R1,
        "center_mode": center_mode,
        "tube_radius_min": radius,
        "n_slabs": float(len(slabs)),
        "n_direct_samples": float(len(sample_df)),
        "z_skeleton": z_skeleton,
        "dR_step": float(dR_step),
        "dE_step": float(dE_step),
        "cauchy_R_step": float(cauchy_R_step),
        "direct_guard_factor": float(direct_guard_factor),
        "sampled_direct_dH_dR_abs_max": float(max(sampled_max_by_slab)),
        "direct_L_bound_global": float(direct_guard_factor) * float(max(sampled_max_by_slab)),
        "s24_interval_L_bound": s24_L,
        "direct_L_global_over_s24_L": float(direct_guard_factor) * float(max(sampled_max_by_slab)) / max(s24_L, 1.0e-300),
        "max_direct_L_over_s24_L": float(slab_df["direct_L_over_s24_L"].max()),
        "max_direct_remainder_over_reserve": float(slab_df["direct_remainder_over_reserve"].max()),
        "median_direct_remainder_over_reserve": float(slab_df["direct_remainder_over_reserve"].median()),
        "direct_integral_ratio": direct_ratio,
        "s24_interval_integral_ratio": s24_ratio,
        "min_dF_dE_abs": float(slab_df["dF_dE_abs_min"].min()),
        "max_dF_dE_CR_rel_mismatch": float(slab_df["dF_dE_CR_rel_mismatch_max"].max()),
        "n_special_interval_pass": n_pass,
        "special_interval_status": "special_interval_certificate_closed"
        if n_pass == len(slabs) and direct_ratio < 1.0
        else "special_interval_certificate_not_closed",
    }
    return sample_df, slab_df, summary


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_parity_segments": float(len(summary)),
                "n_special_interval_closed": float(
                    (summary["special_interval_status"] == "special_interval_certificate_closed").sum()
                ),
                "max_direct_L_global_over_s24_L": float(summary["direct_L_global_over_s24_L"].max()),
                "max_direct_integral_ratio": float(summary["direct_integral_ratio"].max()),
                "median_direct_integral_ratio": float(summary["direct_integral_ratio"].median()),
                "max_direct_remainder_over_reserve": float(summary["max_direct_remainder_over_reserve"].max()),
                "min_dF_dE_abs": float(summary["min_dF_dE_abs"].min()),
                "max_dF_dE_CR_rel_mismatch": float(summary["max_dF_dE_CR_rel_mismatch"].max()),
                "special_interval_status": "special_interval_certificate_closed_on_grid"
                if float(
                    (summary["special_interval_status"] == "special_interval_certificate_closed").sum()
                )
                == len(summary)
                else "special_interval_certificate_not_closed",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Special-function interval certificate for the S24 constants.")
    ap.add_argument("--argument-csv", default=str(DEFAULT_ARGUMENT))
    ap.add_argument("--s22-slabs-csv", default=str(DEFAULT_S22_SLABS))
    ap.add_argument("--s24-summary-csv", default=str(DEFAULT_S24_INTERVALS))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="6")
    ap.add_argument("--R-tail-values", default="400")
    ap.add_argument("--parities", default="even,odd")
    ap.add_argument("--segments", default="60:80")
    ap.add_argument("--center-mode", default="cubic_path")
    ap.add_argument("--z-skeleton", choices=("corners_center", "octant_center"), default="corners_center")
    ap.add_argument("--dR-step", type=float, default=0.5)
    ap.add_argument("--dE-step-fraction", type=float, default=0.001)
    ap.add_argument("--cauchy-R-step-fraction", type=float, default=0.25)
    ap.add_argument("--direct-guard-factor", type=float, default=2.0)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    argument_csv = Path(args.argument_csv)
    s22_slabs_csv = Path(args.s22_slabs_csv)
    s24_summary_csv = Path(args.s24_summary_csv)
    if not argument_csv.exists():
        raise FileNotFoundError(f"Missing argument-principle CSV: {argument_csv}")
    if not s22_slabs_csv.exists():
        raise FileNotFoundError(f"Missing S22 slab CSV: {s22_slabs_csv}")
    if not s24_summary_csv.exists():
        raise FileNotFoundError(f"Missing S24 summary CSV: {s24_summary_csv}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(argument_csv)
    slabs = pd.read_csv(s22_slabs_csv)
    s24 = pd.read_csv(s24_summary_csv)
    Ds = parse_floats(args.Ds)
    Rtails = parse_floats(args.R_tail_values)
    parities = parse_strings(args.parities)
    segments = parse_segments(args.segments)
    if Ds is not None:
        detail = detail[detail["D"].isin(Ds)]
        slabs = slabs[slabs["D"].isin(Ds)]
        s24 = s24[s24["D"].isin(Ds)]
    if Rtails is not None:
        detail = detail[detail["R_tail"].isin(Rtails)]
        slabs = slabs[slabs["R_tail"].isin(Rtails)]
        s24 = s24[s24["R_tail"].isin(Rtails)]
    if parities is not None:
        detail = detail[detail["parity"].isin(parities)]
        slabs = slabs[slabs["parity"].isin(parities)]
        s24 = s24[s24["parity"].isin(parities)]
    if not segments:
        raise ValueError("Need nonempty --segments.")

    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()
    sample_frames: list[pd.DataFrame] = []
    slab_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []

    for (D, R_tail, parity), path in detail.groupby(["D", "R_tail", "parity"], sort=True):
        for R0, R1 in segments:
            slab_group = slabs[
                (slabs["D"] == D)
                & (slabs["R_tail"] == R_tail)
                & (slabs["parity"] == parity)
                & (slabs["R0"] == R0)
                & (slabs["R1"] == R1)
            ].copy()
            s24_group = s24[
                (s24["D"] == D)
                & (s24["R_tail"] == R_tail)
                & (s24["parity"] == parity)
                & (s24["R0"] == R0)
                & (s24["R1"] == R1)
            ].copy()
            if len(slab_group) == 0:
                continue
            if len(s24_group) != 1:
                raise ValueError(f"Need exactly one S24 row for {(D, R_tail, parity, R0, R1)}; got {len(s24_group)}")
            print(
                f"[special-interval] D={D:g} Rc={R_tail:g} parity={parity} "
                f"R={R0:g}->{R1:g}",
                flush=True,
            )
            samples, slab_detail, summary = audit_parity(
                path=path,
                slab_rows=slab_group,
                s24_row=s24_group.iloc[0],
                center_mode=str(args.center_mode),
                z_skeleton=str(args.z_skeleton),
                dR_step=float(args.dR_step),
                dE_step_fraction=float(args.dE_step_fraction),
                cauchy_R_step_fraction=float(args.cauchy_R_step_fraction),
                direct_guard_factor=float(args.direct_guard_factor),
                cfg=cfg,
                p=p,
            )
            sample_frames.append(samples)
            slab_frames.append(slab_detail)
            summaries.append(summary)

    if not summaries:
        raise ValueError("No special interval rows were audited.")

    samples = pd.concat(sample_frames, ignore_index=True)
    slab_detail = pd.concat(slab_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    aggregate = aggregate_summary(summary)

    samples_out = OUTDIR / f"{args.out_tag}_samples.csv"
    slabs_out = OUTDIR / f"{args.out_tag}_slabs.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    aggregate_out = OUTDIR / f"{args.out_tag}_aggregate_summary.csv"
    samples.to_csv(samples_out, index=False)
    slab_detail.to_csv(slabs_out, index=False)
    summary.to_csv(summary_out, index=False)
    aggregate.to_csv(aggregate_out, index=False)

    print(f"Wrote {samples_out}")
    print(f"Wrote {slabs_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {aggregate_out}")
    print("[Special interval summary]")
    print(summary.to_string(index=False))
    print("[Special interval aggregate]")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

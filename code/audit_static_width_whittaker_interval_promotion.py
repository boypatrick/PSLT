#!/usr/bin/env python3
"""
Intervalized parent-promotion audit for the final Whittaker-DtN flowbox.

S23 rewrote the S22 5% slab inflation as a Cauchy derivative remainder.  This
audit makes the node-slope derivative envelope intervalized: instead of using a
separate derivative constant on every slab, the default certificate uses one
parity-uniform constant

    Lambda_pi = guard * max_j |B_pi(R_{j+1}) - B_pi(R_j)| / h_j

over the whole R=60..80 parent interval.  If

    (h_j / 2) Lambda_pi <= 0.05 M_{pi,j}

for all slabs and the integrated beta budget remains below the tube radius,
then the S23 Cauchy/slab proof promotes directly to the parent interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_adapted_flow import parse_floats, parse_strings


DEFAULT_S22_SLABS = OUTDIR / "static_width_whittaker_slab_envelope_d6rc400_r60_r80_c640_slabs.csv"
DEFAULT_TAG = "static_width_whittaker_interval_promotion_d6rc400_r60_r80_c640"


def parse_segments(raw: str) -> tuple[tuple[float, float], ...]:
    if raw.strip() == "":
        return ()
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        left, right = item.strip().split(":")
        out.append((float(left), float(right)))
    return tuple(out)


def source_window(index: int, n_slabs: int, slope_window: str) -> tuple[int, int]:
    if slope_window == "global":
        return 0, n_slabs - 1
    width = int(slope_window)
    if width < 0:
        raise ValueError("--slope-window must be 'global' or a nonnegative integer")
    return max(0, index - width), min(n_slabs - 1, index + width)


def audit_group(
    slab_rows: pd.DataFrame,
    slope_window: str,
    guard_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    slabs = slab_rows.sort_values("slab_index").reset_index(drop=True)
    if slabs.empty:
        raise ValueError("Cannot audit an empty slab group.")

    D = float(slabs["D"].iloc[0])
    R_tail = float(slabs["R_tail"].iloc[0])
    parity = str(slabs["parity"].iloc[0])
    R0 = float(slabs["R0"].iloc[0])
    R1 = float(slabs["R1"].iloc[0])
    center_mode = str(slabs["center_mode"].iloc[0])
    tube_factor = float(slabs["tube_factor"].iloc[0])
    radius = float(slabs["tube_radius_min"].iloc[0])
    n_slabs = len(slabs)

    slab_out: list[dict[str, float | str]] = []
    interval_rows: list[dict[str, float | str]] = []
    seen_intervals: set[tuple[int, int]] = set()

    slopes = slabs["node_slope_abs"].to_numpy(dtype=float)
    for local_index, slab in slabs.iterrows():
        src_left, src_right = source_window(local_index, n_slabs, slope_window)
        source_slopes = slopes[src_left : src_right + 1]
        max_source_slope = float(np.max(source_slopes))
        L_interval = float(guard_factor) * max_source_slope

        h_slab = float(slab["delta_R"])
        M_j = float(slab["node_upper_beta"])
        reserve = 0.05 * M_j
        remainder = 0.5 * h_slab * L_interval
        beta_interval = M_j + remainder
        beta_s22 = float(slab["inflated_beta"])

        if (src_left, src_right) not in seen_intervals:
            seen_intervals.add((src_left, src_right))
            interval_rows.append(
                {
                    "D": D,
                    "R_tail": R_tail,
                    "parity": parity,
                    "R0": R0,
                    "R1": R1,
                    "center_mode": center_mode,
                    "tube_factor": tube_factor,
                    "tube_radius_min": radius,
                    "slope_window": slope_window,
                    "source_slab_left": float(src_left),
                    "source_slab_right": float(src_right),
                    "source_R_left": float(slabs.loc[src_left, "R_left"]),
                    "source_R_right": float(slabs.loc[src_right, "R_right"]),
                    "max_node_slope_abs": max_source_slope,
                    "guard_factor": float(guard_factor),
                    "interval_L_bound": L_interval,
                }
            )

        slab_out.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "tube_factor": tube_factor,
                "tube_radius_min": radius,
                "slab_index": float(slab["slab_index"]),
                "R_left": float(slab["R_left"]),
                "R_right": float(slab["R_right"]),
                "delta_R": h_slab,
                "node_upper_beta": M_j,
                "slope_window": slope_window,
                "source_slab_left": float(src_left),
                "source_slab_right": float(src_right),
                "source_max_node_slope_abs": max_source_slope,
                "guard_factor": float(guard_factor),
                "interval_L_bound": L_interval,
                "five_percent_reserve": reserve,
                "interval_remainder_bound": remainder,
                "interval_remainder_over_reserve": remainder / max(reserve, 1.0e-300),
                "beta_interval": beta_interval,
                "beta_s22_inflated": beta_s22,
                "beta_interval_over_s22_inflated": beta_interval / max(beta_s22, 1.0e-300),
                "interval_contribution": h_slab * beta_interval,
                "s22_inflated_contribution": float(slab["inflated_contribution"]),
                "interval_slab_pass": float(beta_interval <= beta_s22 and remainder <= reserve),
            }
        )

    slab_df = pd.DataFrame(slab_out)
    interval_df = pd.DataFrame(interval_rows)
    interval_ratio = float(slab_df["interval_contribution"].sum() / radius)
    inflated_ratio = float(slab_df["s22_inflated_contribution"].sum() / radius)
    n_pass = float(slab_df["interval_slab_pass"].sum())
    summary = {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R0,
        "R1": R1,
        "center_mode": center_mode,
        "tube_factor": tube_factor,
        "tube_radius_min": radius,
        "n_slabs": float(n_slabs),
        "slope_window": slope_window,
        "guard_factor": float(guard_factor),
        "n_interval_certificates": float(len(interval_df)),
        "max_interval_L_bound": float(slab_df["interval_L_bound"].max()),
        "max_source_node_slope_abs": float(slab_df["source_max_node_slope_abs"].max()),
        "max_interval_remainder_over_reserve": float(slab_df["interval_remainder_over_reserve"].max()),
        "median_interval_remainder_over_reserve": float(slab_df["interval_remainder_over_reserve"].median()),
        "max_beta_interval_over_s22_inflated": float(slab_df["beta_interval_over_s22_inflated"].max()),
        "interval_integral_ratio": interval_ratio,
        "s22_inflated_integral_ratio": inflated_ratio,
        "n_interval_slab_pass": n_pass,
        "interval_promotion_status": "interval_parent_promotion_closed"
        if n_pass == n_slabs and interval_ratio < 1.0
        else "interval_parent_promotion_not_closed",
    }
    return interval_df, slab_df, summary


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_parity_segments": float(len(summary)),
                "n_interval_promoted": float(
                    (summary["interval_promotion_status"] == "interval_parent_promotion_closed").sum()
                ),
                "max_interval_integral_ratio": float(summary["interval_integral_ratio"].max()),
                "median_interval_integral_ratio": float(summary["interval_integral_ratio"].median()),
                "max_interval_remainder_over_reserve": float(
                    summary["max_interval_remainder_over_reserve"].max()
                ),
                "max_beta_interval_over_s22_inflated": float(
                    summary["max_beta_interval_over_s22_inflated"].max()
                ),
                "max_interval_L_bound": float(summary["max_interval_L_bound"].max()),
                "interval_promotion_status": "interval_parent_promotion_closed_on_grid"
                if float(
                    (summary["interval_promotion_status"] == "interval_parent_promotion_closed").sum()
                )
                == len(summary)
                else "interval_parent_promotion_not_closed",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Intervalized parent-promotion audit for S23.")
    ap.add_argument("--s22-slabs-csv", default=str(DEFAULT_S22_SLABS))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="6")
    ap.add_argument("--R-tail-values", default="400")
    ap.add_argument("--parities", default="even,odd")
    ap.add_argument("--segments", default="60:80")
    ap.add_argument("--slope-window", default="global", help="'global' or an integer neighbor window.")
    ap.add_argument("--guard-factor", type=float, default=5.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    s22_slabs_csv = Path(args.s22_slabs_csv)
    if not s22_slabs_csv.exists():
        raise FileNotFoundError(f"Missing S22 slabs CSV: {s22_slabs_csv}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    s22_slabs = pd.read_csv(s22_slabs_csv)
    Ds = parse_floats(args.Ds)
    Rtails = parse_floats(args.R_tail_values)
    parities = parse_strings(args.parities)
    segments = parse_segments(args.segments)
    if Ds is not None:
        s22_slabs = s22_slabs[s22_slabs["D"].isin(Ds)]
    if Rtails is not None:
        s22_slabs = s22_slabs[s22_slabs["R_tail"].isin(Rtails)]
    if parities is not None:
        s22_slabs = s22_slabs[s22_slabs["parity"].isin(parities)]
    if not segments:
        raise ValueError("Need nonempty --segments.")

    interval_frames: list[pd.DataFrame] = []
    slab_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []
    for (D, R_tail, parity), group in s22_slabs.groupby(["D", "R_tail", "parity"], sort=True):
        for R0, R1 in segments:
            slabs = group[(group["R0"] == R0) & (group["R1"] == R1)].copy()
            if len(slabs) == 0:
                continue
            print(
                f"[interval-promotion] D={D:g} Rc={R_tail:g} parity={parity} "
                f"R={R0:g}->{R1:g} window={args.slope_window}",
                flush=True,
            )
            intervals, slab_detail, summary = audit_group(
                slab_rows=slabs,
                slope_window=str(args.slope_window),
                guard_factor=float(args.guard_factor),
            )
            interval_frames.append(intervals)
            slab_frames.append(slab_detail)
            summaries.append(summary)

    if not summaries:
        raise ValueError("No interval-promotion rows were audited.")

    intervals = pd.concat(interval_frames, ignore_index=True)
    slab_detail = pd.concat(slab_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    aggregate = aggregate_summary(summary)

    intervals_out = OUTDIR / f"{args.out_tag}_intervals.csv"
    slabs_out = OUTDIR / f"{args.out_tag}_slabs.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    aggregate_out = OUTDIR / f"{args.out_tag}_aggregate_summary.csv"
    intervals.to_csv(intervals_out, index=False)
    slab_detail.to_csv(slabs_out, index=False)
    summary.to_csv(summary_out, index=False)
    aggregate.to_csv(aggregate_out, index=False)

    print(f"Wrote {intervals_out}")
    print(f"Wrote {slabs_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {aggregate_out}")
    print("[Interval promotion summary]")
    print(summary.to_string(index=False))
    print("[Interval promotion aggregate]")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

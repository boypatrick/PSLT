#!/usr/bin/env python3
"""
Slab-envelope audit for the final Whittaker-DtN static-width flowbox.

This is the S22 follow-up to the parent-scale S21 closure.  It does not solve
new roots, change the Whittaker backend, or introduce a new pole family.  It
takes the already isolated finite-radius Whittaker roots and tests whether the
sampled parent-flowbox envelope on D=6, R_tail=400, R=60->80 can be written in
the reviewer-facing slab form

    sum_j |I_j| beta_j < c rho,

where beta_j is an explicit inflated upper Darboux envelope for

    |F_R + gamma'(R) F_E| / |F_E|

over a moving tube around the same cubic centerline used in S21.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_adapted_flow import (
    derivative_estimates,
    make_eval,
    parse_floats,
    parse_strings,
    root_complex,
    sample_points,
)
from audit_static_width_whittaker_curved_flowbox import polynomial_centerline
from extract_static_width_exact_dtn import ExactDtnConfig
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_slab_envelope_d6rc400_r60_r80_c640"


def parse_segments(raw: str) -> tuple[tuple[float, float], ...]:
    if raw.strip() == "":
        return ()
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        left, right = item.strip().split(":")
        out.append((float(left), float(right)))
    return tuple(out)


def node_grid(n_nodes: int, raw_nodes: str | None) -> tuple[float, ...]:
    if raw_nodes is not None and raw_nodes.strip():
        nodes = parse_floats(raw_nodes)
        if nodes is None:
            raise ValueError("Could not parse --tau-nodes.")
        return nodes
    if n_nodes < 2:
        raise ValueError("--n-r-nodes must be at least 2.")
    return tuple(float(x) for x in np.linspace(0.0, 1.0, int(n_nodes)))


def audit_segment(
    path: pd.DataFrame,
    row_a: pd.Series,
    row_b: pd.Series,
    center_mode: str,
    tube_factor: float,
    tau_nodes: tuple[float, ...],
    n_per_side: int,
    dR_step: float,
    dE_step_fraction: float,
    include_center: bool,
    slab_inflation: float,
    cr_mismatch_max: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    D = float(row_a["D"])
    R_tail = float(row_a["R_tail"])
    parity = str(row_a["parity"])
    ell = int(round(float(row_a.get("ell", cfg.ell))))
    n_half = int(round(float(row_a.get("n_half", cfg.n_half))))
    R0 = float(row_a["R"])
    R1 = float(row_b["R"])
    dR_total = R1 - R0
    if dR_total <= 0:
        raise ValueError(f"Bad segment order: R0={R0}, R1={R1}")

    gamma, gamma_prime, anchors = polynomial_centerline(path=path, R0=R0, R1=R1, mode=center_mode)
    E0 = root_complex(row_a)
    E1 = root_complex(row_b)
    endpoint_error = max(abs(gamma(R0) - E0), abs(gamma(R1) - E1))

    box_re = min(float(row_a["box_re"]), float(row_b["box_re"]))
    box_im = min(float(row_a["box_im"]), float(row_b["box_im"]))
    half_re = float(tube_factor) * box_re
    half_im = float(tube_factor) * box_im
    radius_min = min(half_re, half_im)
    dE_step = max(radius_min * float(dE_step_fraction), 1.0e-8)

    node_rows: list[dict[str, float | str]] = []
    for idx, tau in enumerate(tau_nodes):
        if tau < 0.0 or tau > 1.0:
            raise ValueError(f"Tau node outside [0,1]: {tau}")
        R = R0 + tau * dR_total
        if R - dR_step <= 0.0 or R + dR_step >= R_tail:
            raise ValueError(f"Bad R derivative stencil: R={R}, step={dR_step}, R_tail={R_tail}")

        center = gamma(R)
        speed = gamma_prime(R)
        eval_minus = make_eval(D, R - dR_step, R_tail, parity, n_half, ell, cfg, p)
        eval_mid = make_eval(D, R, R_tail, parity, n_half, ell, cfg, p)
        eval_plus = make_eval(D, R + dR_step, R_tail, parity, n_half, ell, cfg, p)
        pts = sample_points(
            center=center,
            half_re=half_re,
            half_im=half_im,
            n_per_side=n_per_side,
            include_center=include_center,
        )

        q_vals: list[float] = []
        raw_vals: list[float] = []
        denom_vals: list[float] = []
        numerator_vals: list[float] = []
        cr_vals: list[float] = []
        for E in pts:
            deriv = derivative_estimates(
                E=complex(E),
                eval_minus=eval_minus,
                eval_mid=eval_mid,
                eval_plus=eval_plus,
                dR_step=dR_step,
                dE_step=dE_step,
            )
            dF_dR = complex(deriv["dF_dR"])
            dF_dE = complex(deriv["dF_dE"])
            denom = max(abs(dF_dE), 1.0e-300)
            numerator = dF_dR + speed * dF_dE
            q_vals.append(float(abs(numerator) / denom))
            raw_vals.append(float(abs(dF_dR) / denom))
            denom_vals.append(float(abs(dF_dE)))
            numerator_vals.append(float(abs(numerator)))
            cr_vals.append(float(deriv["dF_dE_CR_rel_mismatch"]))

        node_rows.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "center_anchors": ";".join(f"{x:g}" for x in anchors),
                "tube_factor": float(tube_factor),
                "tube_radius_min": radius_min,
                "node_index": float(idx),
                "tau": float(tau),
                "R": float(R),
                "center_real": float(center.real),
                "center_imag": float(center.imag),
                "center_speed_abs": float(abs(speed)),
                "n_points": float(len(pts)),
                "max_q": float(max(q_vals)),
                "median_q": float(np.median(q_vals)),
                "max_raw_flow": float(max(raw_vals)),
                "dF_dE_abs_min": float(min(denom_vals)),
                "dF_dE_abs_median": float(np.median(denom_vals)),
                "curved_numerator_abs_max": float(max(numerator_vals)),
                "dF_dE_CR_rel_mismatch_max": float(max(cr_vals)),
            }
        )

    node_df = pd.DataFrame(node_rows).sort_values("tau").reset_index(drop=True)
    q = node_df["max_q"].to_numpy(dtype=float)
    r_nodes = node_df["R"].to_numpy(dtype=float)
    trapz_integral = float(abs(np.trapezoid(q, x=r_nodes)))

    slab_rows: list[dict[str, float | str]] = []
    upper_integral = 0.0
    inflated_integral = 0.0
    for j in range(len(node_df) - 1):
        left = node_df.iloc[j]
        right = node_df.iloc[j + 1]
        dR = float(right["R"] - left["R"])
        node_max = float(max(left["max_q"], right["max_q"]))
        beta = (1.0 + float(slab_inflation)) * node_max
        upper_integral += dR * node_max
        inflated_integral += dR * beta
        slab_rows.append(
            {
                "D": D,
                "R_tail": R_tail,
                "parity": parity,
                "R0": R0,
                "R1": R1,
                "center_mode": center_mode,
                "tube_factor": float(tube_factor),
                "tube_radius_min": radius_min,
                "slab_index": float(j),
                "R_left": float(left["R"]),
                "R_right": float(right["R"]),
                "delta_R": dR,
                "q_left": float(left["max_q"]),
                "q_right": float(right["max_q"]),
                "node_upper_beta": node_max,
                "inflated_beta": beta,
                "node_upper_contribution": dR * node_max,
                "inflated_contribution": dR * beta,
                "node_slope_abs": abs(float(right["max_q"] - left["max_q"])) / max(dR, 1.0e-300),
            }
        )

    slab_df = pd.DataFrame(slab_rows)
    sup_integral = float(abs(dR_total) * np.max(q))
    radius = max(radius_min, 1.0e-300)
    cr_max = float(node_df["dF_dE_CR_rel_mismatch_max"].max())
    den_min = float(node_df["dF_dE_abs_min"].min())
    pass_gate = (
        inflated_integral / radius < 1.0
        and den_min > 1.0e-10
        and cr_max <= float(cr_mismatch_max)
        and endpoint_error <= 1.0e-10
    )
    summary = {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R0,
        "R1": R1,
        "center_mode": center_mode,
        "center_anchors": ";".join(f"{x:g}" for x in anchors),
        "tube_factor": float(tube_factor),
        "tube_radius_min": radius_min,
        "slab_inflation": float(slab_inflation),
        "n_r_nodes": float(len(node_df)),
        "n_slabs": float(max(len(node_df) - 1, 0)),
        "n_points_per_node": float(len(pts)),
        "dR_step": float(dR_step),
        "dE_step": float(dE_step),
        "endpoint_interpolation_error": float(endpoint_error),
        "max_node_q": float(np.max(q)),
        "median_node_q": float(np.median(q)),
        "min_node_q": float(np.min(q)),
        "sup_integral_bound": sup_integral,
        "trapz_integral_bound": trapz_integral,
        "upper_darboux_integral_bound": float(upper_integral),
        "inflated_slab_integral_bound": float(inflated_integral),
        "sup_contraction_ratio": sup_integral / radius,
        "trapz_contraction_ratio": trapz_integral / radius,
        "upper_darboux_contraction_ratio": float(upper_integral) / radius,
        "inflated_slab_contraction_ratio": float(inflated_integral) / radius,
        "dF_dE_abs_min": den_min,
        "dF_dE_abs_median_min": float(node_df["dF_dE_abs_median"].min()),
        "curved_numerator_abs_max": float(node_df["curved_numerator_abs_max"].max()),
        "dF_dE_CR_rel_mismatch_max": cr_max,
        "max_node_slope_abs": float(slab_df["node_slope_abs"].max()) if len(slab_df) else 0.0,
        "slab_envelope_pass": float(pass_gate),
        "slab_envelope_status": "slab_envelope_closed" if pass_gate else "slab_envelope_not_closed",
    }
    return node_df, slab_df, summary


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_segments": float(len(summary)),
                "n_slab_envelope_pass": float(summary["slab_envelope_pass"].sum()),
                "max_inflated_slab_contraction_ratio": float(summary["inflated_slab_contraction_ratio"].max()),
                "median_inflated_slab_contraction_ratio": float(summary["inflated_slab_contraction_ratio"].median()),
                "max_upper_darboux_contraction_ratio": float(summary["upper_darboux_contraction_ratio"].max()),
                "max_trapz_contraction_ratio": float(summary["trapz_contraction_ratio"].max()),
                "min_dF_dE_abs": float(summary["dF_dE_abs_min"].min()),
                "max_dF_dE_CR_rel_mismatch": float(summary["dF_dE_CR_rel_mismatch_max"].max()),
                "slab_envelope_status": "slab_envelope_closed_on_grid"
                if float(summary["slab_envelope_pass"].sum()) == len(summary)
                else "slab_envelope_not_closed",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Slab-envelope audit for the final Whittaker-DtN static-width flowbox.")
    ap.add_argument("--argument-csv", default=str(DEFAULT_ARGUMENT))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="6")
    ap.add_argument("--R-tail-values", default="400")
    ap.add_argument("--parities", default="even,odd")
    ap.add_argument("--segments", default="60:80")
    ap.add_argument("--center-mode", default="cubic_path")
    ap.add_argument("--tube-factor", type=float, default=640.0)
    ap.add_argument("--n-r-nodes", type=int, default=33)
    ap.add_argument("--tau-nodes", default="")
    ap.add_argument("--n-per-side", type=int, default=2)
    ap.add_argument("--dR-step", type=float, default=0.5)
    ap.add_argument("--dE-step-fraction", type=float, default=0.001)
    ap.add_argument("--slab-inflation", type=float, default=0.05)
    ap.add_argument("--cr-mismatch-max", type=float, default=5.0e-2)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--no-center", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    argument_csv = Path(args.argument_csv)
    if not argument_csv.exists():
        raise FileNotFoundError(f"Missing argument-principle CSV: {argument_csv}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(argument_csv)
    Ds = parse_floats(args.Ds)
    Rtails = parse_floats(args.R_tail_values)
    parities = parse_strings(args.parities)
    segments = parse_segments(args.segments)
    tau_nodes = node_grid(args.n_r_nodes, args.tau_nodes)
    if Ds is not None:
        detail = detail[detail["D"].isin(Ds)]
    if Rtails is not None:
        detail = detail[detail["R_tail"].isin(Rtails)]
    if parities is not None:
        detail = detail[detail["parity"].isin(parities)]
    if not segments:
        raise ValueError("Need nonempty --segments.")

    detail = detail.sort_values(["D", "R_tail", "parity", "R"]).reset_index(drop=True)
    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()

    node_frames: list[pd.DataFrame] = []
    slab_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    for (D, R_tail, parity), path in detail.groupby(["D", "R_tail", "parity"], sort=True):
        path = path.sort_values("R").reset_index(drop=True)
        for R0, R1 in segments:
            selected = path[path["R"].isin([R0, R1])].sort_values("R")
            if len(selected) != 2:
                continue
            print(f"[slab-envelope] D={D:g} Rc={R_tail:g} parity={parity} R={R0:g}->{R1:g}", flush=True)
            node_df, slab_df, summary = audit_segment(
                path=path,
                row_a=selected.iloc[0],
                row_b=selected.iloc[1],
                center_mode=str(args.center_mode),
                tube_factor=float(args.tube_factor),
                tau_nodes=tau_nodes,
                n_per_side=int(args.n_per_side),
                dR_step=float(args.dR_step),
                dE_step_fraction=float(args.dE_step_fraction),
                include_center=not bool(args.no_center),
                slab_inflation=float(args.slab_inflation),
                cr_mismatch_max=float(args.cr_mismatch_max),
                cfg=cfg,
                p=p,
            )
            node_frames.append(node_df)
            slab_frames.append(slab_df)
            summary_rows.append(summary)

    if not summary_rows:
        raise ValueError("No slab-envelope rows were audited.")
    node_detail = pd.concat(node_frames, ignore_index=True)
    slab_detail = pd.concat(slab_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    aggregate = aggregate_summary(summary)

    node_out = OUTDIR / f"{args.out_tag}_nodes.csv"
    slab_out = OUTDIR / f"{args.out_tag}_slabs.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    aggregate_out = OUTDIR / f"{args.out_tag}_aggregate_summary.csv"
    node_detail.to_csv(node_out, index=False)
    slab_detail.to_csv(slab_out, index=False)
    summary.to_csv(summary_out, index=False)
    aggregate.to_csv(aggregate_out, index=False)

    print(f"Wrote {node_out}")
    print(f"Wrote {slab_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {aggregate_out}")
    print("[Slab-envelope summary]")
    print(summary.to_string(index=False))
    print("[Slab-envelope aggregate]")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

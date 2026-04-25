#!/usr/bin/env python3
"""
Curvature-adapted early-R flowbox audit for the Whittaker-DtN static-width gate.

This is the narrow follow-up to the S19 adapted-secant audit.  It keeps the
same finite-radius Whittaker-DtN residual and the same already-isolated roots,
but replaces the straight secant centerline by parent-side polynomial
centerlines built from existing R anchors:

    gamma(R) in {secant, one-sided quadratic, full four-anchor cubic}.

For any such centerline with gamma(R_a)=E_a and gamma(R_b)=E_b, the branch
deviation Z=E-gamma satisfies

    Z' = -(F_R + gamma'(R) F_E) / F_E.

The certified gate is therefore

    (R_b-R_a) sup |F_R + gamma'(R) F_E| / |F_E| < tube_radius.

No new roots are solved, no backend is changed, and no new resonance family is
introduced.  The default target is only the early S19 bottleneck:
D=6, R_tail=400, R=60->80 and 80->100.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_curved_flowbox_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_curved_flowbox_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_curved_flowbox_adaptive_summary.csv
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
from extract_static_width_exact_dtn import ExactDtnConfig
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_curved_flowbox"


def parse_segments(raw: str) -> tuple[tuple[float, float], ...]:
    if raw.strip() == "":
        return ()
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        left, right = item.strip().split(":")
        out.append((float(left), float(right)))
    return tuple(out)


def polynomial_centerline(
    path: pd.DataFrame,
    R0: float,
    R1: float,
    mode: str,
) -> tuple[callable, callable, tuple[float, ...]]:
    """Return gamma, gamma', and the anchor R values for a centerline mode."""
    path = path.sort_values("R").reset_index(drop=True)
    available = tuple(float(x) for x in path["R"].to_numpy(dtype=float))
    if mode == "secant":
        rows = path[path["R"].isin([R0, R1])].sort_values("R")
    elif mode == "quadratic_forward":
        later = [x for x in available if x > R1]
        if not later:
            raise ValueError("quadratic_forward needs one anchor after R1")
        rows = path[path["R"].isin([R0, R1, min(later)])].sort_values("R")
    elif mode == "quadratic_backward":
        earlier = [x for x in available if x < R0]
        if not earlier:
            raise ValueError("quadratic_backward needs one anchor before R0")
        rows = path[path["R"].isin([max(earlier), R0, R1])].sort_values("R")
    elif mode == "cubic_path":
        if len(path) < 4:
            raise ValueError("cubic_path needs four path anchors")
        rows = path
    else:
        raise ValueError(f"Unknown centerline mode: {mode}")

    anchors = tuple(float(x) for x in rows["R"].to_numpy(dtype=float))
    values = np.asarray([root_complex(row) for _, row in rows.iterrows()], dtype=complex)
    degree = min(len(anchors) - 1, 3)
    coeff = np.polyfit(np.asarray(anchors, dtype=float), values, degree)
    dcoeff = np.polyder(coeff)

    def gamma(R: float) -> complex:
        return complex(np.polyval(coeff, float(R)))

    def gamma_prime(R: float) -> complex:
        return complex(np.polyval(dcoeff, float(R)))

    return gamma, gamma_prime, anchors


def audit_segment(
    path: pd.DataFrame,
    row_a: pd.Series,
    row_b: pd.Series,
    center_mode: str,
    tube_factor: float,
    quad_nodes: tuple[float, ...],
    n_per_side: int,
    dR_step: float,
    dE_step_fraction: float,
    include_center: bool,
    cr_mismatch_max: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    D = float(row_a["D"])
    R_tail = float(row_a["R_tail"])
    parity = str(row_a["parity"])
    ell = int(round(float(row_a.get("ell", cfg.ell))))
    n_half = int(round(float(row_a.get("n_half", cfg.n_half))))
    R0 = float(row_a["R"])
    R1 = float(row_b["R"])
    dR_total = R1 - R0
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

    node_rows: list[dict[str, float]] = []
    for tau in quad_nodes:
        tau = float(tau)
        R_mid = R0 + tau * dR_total
        if R_mid - dR_step <= 0.0 or R_mid + dR_step >= R_tail:
            raise ValueError(f"Bad R derivative stencil: R={R_mid}, step={dR_step}, R_tail={R_tail}")
        center = gamma(R_mid)
        center_speed = gamma_prime(R_mid)
        eval_minus = make_eval(D, R_mid - dR_step, R_tail, parity, n_half, ell, cfg, p)
        eval_mid = make_eval(D, R_mid, R_tail, parity, n_half, ell, cfg, p)
        eval_plus = make_eval(D, R_mid + dR_step, R_tail, parity, n_half, ell, cfg, p)
        pts = sample_points(center=center, half_re=half_re, half_im=half_im, n_per_side=n_per_side, include_center=include_center)

        flow_vals: list[float] = []
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
            numerator = dF_dR + center_speed * dF_dE
            flow_vals.append(float(abs(numerator) / denom))
            raw_vals.append(float(abs(dF_dR) / denom))
            denom_vals.append(float(abs(dF_dE)))
            numerator_vals.append(float(abs(numerator)))
            cr_vals.append(float(deriv["dF_dE_CR_rel_mismatch"]))
        node_rows.append(
            {
                "tau": tau,
                "center_speed_abs": float(abs(center_speed)),
                "max_curved_flow": float(max(flow_vals)),
                "median_curved_flow": float(np.median(flow_vals)),
                "max_raw_flow": float(max(raw_vals)),
                "dF_dE_abs_min": float(min(denom_vals)),
                "dF_dE_abs_median": float(np.median(denom_vals)),
                "curved_numerator_abs_max": float(max(numerator_vals)),
                "dF_dE_CR_rel_mismatch_max": float(max(cr_vals)),
            }
        )

    nodes = pd.DataFrame(node_rows)
    nodes = nodes.sort_values("tau").reset_index(drop=True)
    R_nodes = R0 + nodes["tau"].to_numpy(dtype=float) * dR_total
    curved_node_max = nodes["max_curved_flow"].to_numpy(dtype=float)
    raw_node_max = nodes["max_raw_flow"].to_numpy(dtype=float)
    max_curved_flow = float(np.max(curved_node_max))
    max_raw_flow = float(nodes["max_raw_flow"].max())
    sup_integral_bound = abs(dR_total) * max_curved_flow
    raw_sup_integral_bound = abs(dR_total) * max_raw_flow
    # This is the useful S20 tightening: it audits the integral flow budget
    # directly instead of multiplying the worst sampled R-node by the whole
    # segment length.  A parent proof would replace the nodal envelope by an
    # analytic slab-wise upper envelope.
    trapz_integral_bound = float(abs(np.trapezoid(curved_node_max, x=R_nodes)))
    raw_trapz_integral_bound = float(abs(np.trapezoid(raw_node_max, x=R_nodes)))
    if len(nodes) == 3 and set(np.round(nodes["tau"].to_numpy(dtype=float), 12)) == {0.0, 0.5, 1.0}:
        simpson_integral_bound = float(abs(dR_total) * (curved_node_max[0] + 4.0 * curved_node_max[1] + curved_node_max[2]) / 6.0)
        raw_simpson_integral_bound = float(abs(dR_total) * (raw_node_max[0] + 4.0 * raw_node_max[1] + raw_node_max[2]) / 6.0)
    else:
        simpson_integral_bound = np.nan
        raw_simpson_integral_bound = np.nan
    sup_contraction_ratio = sup_integral_bound / max(radius_min, 1.0e-300)
    trapz_contraction_ratio = trapz_integral_bound / max(radius_min, 1.0e-300)
    simpson_contraction_ratio = simpson_integral_bound / max(radius_min, 1.0e-300) if np.isfinite(simpson_integral_bound) else np.nan
    raw_contraction_ratio = raw_trapz_integral_bound / max(radius_min, 1.0e-300)
    cr_max = float(nodes["dF_dE_CR_rel_mismatch_max"].max())
    pass_gate = (
        trapz_contraction_ratio < 1.0
        and float(nodes["dF_dE_abs_min"].min()) > 1.0e-10
        and cr_max <= float(cr_mismatch_max)
        and endpoint_error <= 1.0e-10
    )
    return {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R0,
        "R1": R1,
        "center_mode": center_mode,
        "center_anchors": ";".join(f"{x:g}" for x in anchors),
        "tube_factor": float(tube_factor),
        "tube_half_re": half_re,
        "tube_half_im": half_im,
        "tube_radius_min": radius_min,
        "n_quad_nodes": float(len(quad_nodes)),
        "n_points_per_node": float(4 * max(4, int(n_per_side)) + (1 if include_center else 0)),
        "dR_step": float(dR_step),
        "dE_step": float(dE_step),
        "endpoint_interpolation_error": float(endpoint_error),
        "center_speed_abs_max": float(nodes["center_speed_abs"].max()),
        "max_curved_flow_integrand": max_curved_flow,
        "median_node_curved_flow_integrand": float(nodes["max_curved_flow"].median()),
        "max_raw_flow_integrand": max_raw_flow,
        "sup_integral_curved_bound": sup_integral_bound,
        "trapz_integral_curved_bound": trapz_integral_bound,
        "simpson_integral_curved_bound": simpson_integral_bound,
        "sup_integral_raw_bound": raw_sup_integral_bound,
        "trapz_integral_raw_bound": raw_trapz_integral_bound,
        "simpson_integral_raw_bound": raw_simpson_integral_bound,
        "sup_curved_contraction_ratio": sup_contraction_ratio,
        "trapz_curved_contraction_ratio": trapz_contraction_ratio,
        "simpson_curved_contraction_ratio": simpson_contraction_ratio,
        "curved_contraction_ratio": trapz_contraction_ratio,
        "raw_contraction_ratio": raw_contraction_ratio,
        "dF_dE_abs_min": float(nodes["dF_dE_abs_min"].min()),
        "dF_dE_abs_median_min": float(nodes["dF_dE_abs_median"].min()),
        "curved_numerator_abs_max": float(nodes["curved_numerator_abs_max"].max()),
        "dF_dE_CR_rel_mismatch_max": cr_max,
        "curved_flow_pass": float(pass_gate),
        "curved_flow_status": "curved_flowbox_closed" if pass_gate else "curved_flowbox_not_closed",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (mode, factor), sub in detail.groupby(["center_mode", "tube_factor"], sort=True):
        rows.append(
            {
                "center_mode": str(mode),
                "tube_factor": float(factor),
                "n_segments": float(len(sub)),
                "n_curved_flow_pass": float(sub["curved_flow_pass"].sum()),
                "max_curved_contraction_ratio": float(sub["curved_contraction_ratio"].max()),
                "median_curved_contraction_ratio": float(sub["curved_contraction_ratio"].median()),
                "min_curved_contraction_ratio": float(sub["curved_contraction_ratio"].min()),
                "max_raw_contraction_ratio": float(sub["raw_contraction_ratio"].max()),
                "min_dF_dE_abs": float(sub["dF_dE_abs_min"].min()),
                "max_dF_dE_CR_rel_mismatch": float(sub["dF_dE_CR_rel_mismatch_max"].max()),
                "curved_flow_status": "curved_flowbox_closed_on_grid"
                if float(sub["curved_flow_pass"].sum()) == len(sub)
                else "curved_flowbox_not_closed",
            }
        )
    return pd.DataFrame(rows)


def adaptive_summarize(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["D", "R_tail", "parity", "R0", "R1"]
    rows: list[dict[str, float | str]] = []
    for key, sub in detail.groupby(keys, sort=True):
        best = sub.loc[sub["curved_contraction_ratio"].idxmin()]
        passes = sub[sub["curved_flow_pass"] > 0.5]
        best_pass = passes.loc[passes["curved_contraction_ratio"].idxmin()] if len(passes) else None
        rows.append(
            {
                **dict(zip(keys, key)),
                "has_curved_flow_pass": float(best_pass is not None),
                "best_center_mode": str(best["center_mode"]),
                "best_tube_factor": float(best["tube_factor"]),
                "best_curved_contraction_ratio": float(best["curved_contraction_ratio"]),
                "best_raw_contraction_ratio": float(best["raw_contraction_ratio"]),
                "best_dF_dE_abs_min": float(best["dF_dE_abs_min"]),
                "best_cr_mismatch": float(best["dF_dE_CR_rel_mismatch_max"]),
                "best_pass_center_mode": "" if best_pass is None else str(best_pass["center_mode"]),
                "best_pass_tube_factor": np.nan if best_pass is None else float(best_pass["tube_factor"]),
                "best_pass_contraction_ratio": np.nan if best_pass is None else float(best_pass["curved_contraction_ratio"]),
            }
        )
    best_df = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "n_segments": float(len(best_df)),
                "n_segments_with_curved_flow_pass": float(best_df["has_curved_flow_pass"].sum()),
                "max_best_curved_contraction_ratio": float(best_df["best_curved_contraction_ratio"].max()),
                "median_best_curved_contraction_ratio": float(best_df["best_curved_contraction_ratio"].median()),
                "min_best_dF_dE_abs": float(best_df["best_dF_dE_abs_min"].min()),
                "max_best_cr_mismatch": float(best_df["best_cr_mismatch"].max()),
                "curved_flow_status": "adaptive_curved_flowbox_closed_on_grid"
                if float(best_df["has_curved_flow_pass"].sum()) == len(best_df)
                else "adaptive_curved_flowbox_not_closed",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Curved centerline flowbox audit for early Whittaker-DtN R segments.")
    ap.add_argument("--argument-csv", default=str(DEFAULT_ARGUMENT))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="6")
    ap.add_argument("--R-tail-values", default="400")
    ap.add_argument("--parities", default="even,odd")
    ap.add_argument("--segments", default="60:80,80:100")
    ap.add_argument("--center-modes", default="secant,quadratic_forward,quadratic_backward,cubic_path")
    ap.add_argument("--tube-factors", default="32,48,64,80,96,128")
    ap.add_argument("--quad-nodes", default="0,0.5,1")
    ap.add_argument("--n-per-side", type=int, default=2)
    ap.add_argument("--dR-step", type=float, default=0.5)
    ap.add_argument("--dE-step-fraction", type=float, default=0.005)
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
    center_modes = parse_strings(args.center_modes)
    tube_factors = parse_floats(args.tube_factors)
    quad_nodes = parse_floats(args.quad_nodes)
    if Ds is not None:
        detail = detail[detail["D"].isin(Ds)]
    if Rtails is not None:
        detail = detail[detail["R_tail"].isin(Rtails)]
    if parities is not None:
        detail = detail[detail["parity"].isin(parities)]
    if not segments or center_modes is None or tube_factors is None or quad_nodes is None:
        raise ValueError("Need nonempty --segments, --center-modes, --tube-factors, and --quad-nodes.")
    for node in quad_nodes:
        if node < 0.0 or node > 1.0:
            raise ValueError(f"Quadrature node outside [0,1]: {node}")

    detail = detail.sort_values(["D", "R_tail", "parity", "R"]).reset_index(drop=True)
    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()
    rows: list[dict[str, float | str]] = []
    groups = list(detail.groupby(["D", "R_tail", "parity"], sort=True))
    for mode in center_modes:
        for tube_factor in tube_factors:
            print(f"[mode={mode} tube={tube_factor:g}] auditing {len(groups)} paths", flush=True)
            for (D, R_tail, parity), path in groups:
                path = path.sort_values("R").reset_index(drop=True)
                for R0, R1 in segments:
                    selected = path[path["R"].isin([R0, R1])].sort_values("R")
                    if len(selected) != 2:
                        continue
                    try:
                        rows.append(
                            audit_segment(
                                path=path,
                                row_a=selected.iloc[0],
                                row_b=selected.iloc[1],
                                center_mode=str(mode),
                                tube_factor=float(tube_factor),
                                quad_nodes=quad_nodes,
                                n_per_side=int(args.n_per_side),
                                dR_step=float(args.dR_step),
                                dE_step_fraction=float(args.dE_step_fraction),
                                include_center=not bool(args.no_center),
                                cr_mismatch_max=float(args.cr_mismatch_max),
                                cfg=cfg,
                                p=p,
                            )
                        )
                    except ValueError as exc:
                        print(f"  skip D={D:g} Rc={R_tail:g} {parity} R={R0:g}->{R1:g}: {exc}", flush=True)

    out_detail = pd.DataFrame(rows)
    if len(out_detail) == 0:
        raise ValueError("No curved-flowbox rows were audited.")
    out_summary = summarize(out_detail)
    out_adaptive = adaptive_summarize(out_detail)

    detail_out = OUTDIR / f"{args.out_tag}_detail.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    adaptive_out = OUTDIR / f"{args.out_tag}_adaptive_summary.csv"
    out_detail.to_csv(detail_out, index=False)
    out_summary.to_csv(summary_out, index=False)
    out_adaptive.to_csv(adaptive_out, index=False)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {adaptive_out}")
    print("[Curved-flowbox summary]")
    print(out_summary.to_string(index=False))
    print("[Adaptive curved-flowbox summary]")
    print(out_adaptive.to_string(index=False))


if __name__ == "__main__":
    main()

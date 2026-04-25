#!/usr/bin/env python3
"""
Adapted-contour implicit-flow audit for the static-width R-homotopy gate.

This is the S19 diagnostic after the large fixed-rectangle Rouché audit.  It
does not solve for new roots and does not change the Whittaker-DtN family.
Instead of asking whether two endpoint residuals are close on one fixed
rectangle, it follows the observed secant between adjacent roots and estimates
the deviation of the implicit vector field from that secant:

    E'(R) = -F_R(R,E) / F_E(R,E),
    Z(R) = E(R) - (E_a + (R-R_a) v_ab),
    Z'(R) = -F_R/F_E - v_ab.

On a moving tube around the secant center, the certificate is

    int | -F_R/F_E - v_ab | dR < tube_radius.

Equivalently, the sampled bound uses

    | -F_R/F_E - v_ab |
      <= |F_R + v_ab F_E| / |F_E|.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_adapted_flow_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_adapted_flow_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_adapted_flow_adaptive_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_argument_principle import (
    contour_rectangle,
    residual_evaluator,
)
from extract_static_width_exact_dtn import ExactDtnConfig
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_adapted_flow"


def parse_floats(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def parse_strings(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def root_complex(row: pd.Series) -> complex:
    return complex(float(row["E0_real"]), float(row["E0_imag"]))


def make_eval(
    D: float,
    R: float,
    R_tail: float,
    parity: str,
    n_half: int,
    ell: int,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> callable:
    eval_f, _ = residual_evaluator(
        D=D,
        R=R,
        R_tail=R_tail,
        parity=parity,
        n_half=n_half,
        ell=ell,
        cfg=replace(cfg, ell=ell, n_half=n_half),
        p=p,
    )
    return eval_f


def derivative_estimates(
    E: complex,
    eval_minus: callable,
    eval_mid: callable,
    eval_plus: callable,
    dR_step: float,
    dE_step: float,
) -> dict[str, complex | float]:
    dR = (eval_plus(E) - eval_minus(E)) / (2.0 * dR_step)
    dE_real = (eval_mid(E + dE_step) - eval_mid(E - dE_step)) / (2.0 * dE_step)
    dE_imag = (eval_mid(E + 1j * dE_step) - eval_mid(E - 1j * dE_step)) / (2.0j * dE_step)
    dE = 0.5 * (dE_real + dE_imag)
    dE_abs = 0.5 * (abs(dE_real) + abs(dE_imag))
    cr_mismatch = abs(dE_real - dE_imag) / max(dE_abs, 1.0e-300)
    return {
        "dF_dR": dR,
        "dF_dE": dE,
        "dF_dE_abs": float(abs(dE)),
        "dF_dE_abs_avg": float(dE_abs),
        "dF_dE_CR_rel_mismatch": float(cr_mismatch),
    }


def sample_points(center: complex, half_re: float, half_im: float, n_per_side: int, include_center: bool) -> np.ndarray:
    pts = contour_rectangle(center=center, half_re=half_re, half_im=half_im, n_per_side=n_per_side)
    if include_center:
        pts = np.concatenate([pts, np.asarray([center], dtype=complex)])
    return pts


def audit_segment(
    row_a: pd.Series,
    row_b: pd.Series,
    tube_factor: float,
    quad_nodes: tuple[float, ...],
    n_per_side: int,
    dR_step: float,
    dE_step_fraction: float,
    include_center: bool,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    D = float(row_a["D"])
    R_tail = float(row_a["R_tail"])
    parity = str(row_a["parity"])
    ell = int(round(float(row_a.get("ell", cfg.ell))))
    n_half = int(round(float(row_a.get("n_half", cfg.n_half))))
    R_a = float(row_a["R"])
    R_b = float(row_b["R"])
    dR_total = R_b - R_a
    E_a = root_complex(row_a)
    E_b = root_complex(row_b)
    v_secant = (E_b - E_a) / dR_total
    box_re = min(float(row_a["box_re"]), float(row_b["box_re"]))
    box_im = min(float(row_a["box_im"]), float(row_b["box_im"]))
    half_re = float(tube_factor) * box_re
    half_im = float(tube_factor) * box_im
    radius_min = min(half_re, half_im)
    dE_step = max(radius_min * float(dE_step_fraction), 1.0e-8)

    node_rows: list[dict[str, float]] = []
    for tau in quad_nodes:
        tau = float(tau)
        R_mid = R_a + tau * dR_total
        center = E_a + tau * (E_b - E_a)
        if R_mid - dR_step <= 0.0 or R_mid + dR_step >= R_tail:
            raise ValueError(f"Bad R derivative stencil: R={R_mid}, step={dR_step}, R_tail={R_tail}")
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
            numerator = dF_dR + v_secant * dF_dE
            flow_vals.append(float(abs(numerator) / denom))
            raw_vals.append(float(abs(dF_dR) / denom))
            denom_vals.append(float(abs(dF_dE)))
            numerator_vals.append(float(abs(numerator)))
            cr_vals.append(float(deriv["dF_dE_CR_rel_mismatch"]))
        node_rows.append(
            {
                "tau": tau,
                "max_centered_flow": float(max(flow_vals)),
                "median_centered_flow": float(np.median(flow_vals)),
                "max_raw_flow": float(max(raw_vals)),
                "dF_dE_abs_min": float(min(denom_vals)),
                "dF_dE_abs_median": float(np.median(denom_vals)),
                "centered_numerator_abs_max": float(max(numerator_vals)),
                "dF_dE_CR_rel_mismatch_max": float(max(cr_vals)),
            }
        )

    nodes = pd.DataFrame(node_rows).sort_values("tau")
    max_centered_flow = float(nodes["max_centered_flow"].max())
    max_raw_flow = float(nodes["max_raw_flow"].max())
    integral_bound = abs(dR_total) * max_centered_flow
    raw_integral_bound = abs(dR_total) * max_raw_flow
    contraction_ratio = integral_bound / max(radius_min, 1.0e-300)
    raw_contraction_ratio = raw_integral_bound / max(radius_min, 1.0e-300)
    pass_gate = (
        contraction_ratio < 1.0
        and float(nodes["dF_dE_abs_min"].min()) > 1.0e-10
        and float(nodes["dF_dE_CR_rel_mismatch_max"].max()) < 5.0e-2
    )
    return {
        "D": D,
        "R_tail": R_tail,
        "parity": parity,
        "R0": R_a,
        "R1": R_b,
        "tube_factor": float(tube_factor),
        "tube_half_re": half_re,
        "tube_half_im": half_im,
        "tube_radius_min": radius_min,
        "n_quad_nodes": float(len(quad_nodes)),
        "n_points_per_node": float(4 * max(4, int(n_per_side)) + (1 if include_center else 0)),
        "dR_step": float(dR_step),
        "dE_step": float(dE_step),
        "secant_speed_abs": float(abs(v_secant)),
        "max_centered_flow_integrand": max_centered_flow,
        "median_node_centered_flow_integrand": float(nodes["max_centered_flow"].median()),
        "max_raw_flow_integrand": max_raw_flow,
        "integral_centered_bound": integral_bound,
        "integral_raw_bound": raw_integral_bound,
        "centered_contraction_ratio": contraction_ratio,
        "raw_contraction_ratio": raw_contraction_ratio,
        "dF_dE_abs_min": float(nodes["dF_dE_abs_min"].min()),
        "dF_dE_abs_median_min": float(nodes["dF_dE_abs_median"].min()),
        "centered_numerator_abs_max": float(nodes["centered_numerator_abs_max"].max()),
        "dF_dE_CR_rel_mismatch_max": float(nodes["dF_dE_CR_rel_mismatch_max"].max()),
        "adapted_flow_pass": float(pass_gate),
        "adapted_flow_status": "adapted_secant_flow_closed" if pass_gate else "adapted_secant_flow_not_closed",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for factor, sub in detail.groupby("tube_factor", sort=True):
        rows.append(
            {
                "tube_factor": float(factor),
                "n_segments": float(len(sub)),
                "n_adapted_flow_pass": float(sub["adapted_flow_pass"].sum()),
                "max_centered_contraction_ratio": float(sub["centered_contraction_ratio"].max()),
                "median_centered_contraction_ratio": float(sub["centered_contraction_ratio"].median()),
                "min_centered_contraction_ratio": float(sub["centered_contraction_ratio"].min()),
                "max_raw_contraction_ratio": float(sub["raw_contraction_ratio"].max()),
                "median_raw_contraction_ratio": float(sub["raw_contraction_ratio"].median()),
                "min_dF_dE_abs": float(sub["dF_dE_abs_min"].min()),
                "max_dF_dE_CR_rel_mismatch": float(sub["dF_dE_CR_rel_mismatch_max"].max()),
                "adapted_flow_status": "adapted_secant_flow_closed_on_grid"
                if float(sub["adapted_flow_pass"].sum()) == len(sub)
                else "adapted_secant_flow_not_closed",
            }
        )
    return pd.DataFrame(rows)


def adaptive_summarize(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["D", "R_tail", "parity", "R0", "R1"]
    rows: list[dict[str, float | str]] = []
    for key, sub in detail.groupby(keys, sort=True):
        best = sub.loc[sub["centered_contraction_ratio"].idxmin()]
        passes = sub[sub["adapted_flow_pass"] > 0.5]
        best_pass = passes.loc[passes["centered_contraction_ratio"].idxmin()] if len(passes) else None
        rows.append(
            {
                **dict(zip(keys, key)),
                "has_adapted_flow_pass": float(best_pass is not None),
                "best_tube_factor": float(best["tube_factor"]),
                "best_centered_contraction_ratio": float(best["centered_contraction_ratio"]),
                "best_raw_contraction_ratio": float(best["raw_contraction_ratio"]),
                "best_dF_dE_abs_min": float(best["dF_dE_abs_min"]),
                "best_pass_tube_factor": np.nan if best_pass is None else float(best_pass["tube_factor"]),
                "best_pass_contraction_ratio": np.nan if best_pass is None else float(best_pass["centered_contraction_ratio"]),
            }
        )
    best_df = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "n_segments": float(len(best_df)),
                "n_segments_with_adapted_flow_pass": float(best_df["has_adapted_flow_pass"].sum()),
                "max_best_centered_contraction_ratio": float(best_df["best_centered_contraction_ratio"].max()),
                "median_best_centered_contraction_ratio": float(best_df["best_centered_contraction_ratio"].median()),
                "min_best_dF_dE_abs": float(best_df["best_dF_dE_abs_min"].min()),
                "adapted_flow_status": "adaptive_adapted_flow_closed_on_grid"
                if float(best_df["has_adapted_flow_pass"].sum()) == len(best_df)
                else "adaptive_adapted_flow_not_closed",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Adapted secant implicit-flow audit for Whittaker-DtN roots.")
    ap.add_argument("--argument-csv", default=str(DEFAULT_ARGUMENT))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="", help="Optional comma-separated D filter.")
    ap.add_argument("--R-tail-values", default="", help="Optional comma-separated R_tail filter.")
    ap.add_argument("--parities", default="", help="Optional comma-separated parity filter.")
    ap.add_argument("--tube-factors", default="1,2,4,8", help="Tube half-widths in S8 box units.")
    ap.add_argument("--quad-nodes", default="0,0.5,1", help="Segment quadrature/sample nodes in [0,1].")
    ap.add_argument("--n-per-side", type=int, default=3)
    ap.add_argument("--dR-step", type=float, default=0.5)
    ap.add_argument("--dE-step-fraction", type=float, default=0.10)
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
    if Ds is not None:
        detail = detail[detail["D"].isin(Ds)]
    if Rtails is not None:
        detail = detail[detail["R_tail"].isin(Rtails)]
    if parities is not None:
        detail = detail[detail["parity"].isin(parities)]
    detail = detail.sort_values(["D", "R_tail", "parity", "R"]).reset_index(drop=True)
    if len(detail) == 0:
        raise ValueError("No rows selected for adapted-flow audit.")

    tube_factors = parse_floats(args.tube_factors)
    quad_nodes = parse_floats(args.quad_nodes)
    if tube_factors is None or quad_nodes is None:
        raise ValueError("Need nonempty --tube-factors and --quad-nodes.")
    for node in quad_nodes:
        if node < 0.0 or node > 1.0:
            raise ValueError(f"Quadrature node outside [0,1]: {node}")

    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()
    rows: list[dict[str, float | str]] = []
    groups = list(detail.groupby(["D", "R_tail", "parity"], sort=True))
    for tube_factor in tube_factors:
        print(f"[tube={tube_factor:g}] auditing {len(groups)} paths", flush=True)
        for (D, R_tail, parity), sub in groups:
            sub = sub.sort_values("R").reset_index(drop=True)
            for idx in range(len(sub) - 1):
                print(
                    f"  D={D:g} Rc={R_tail:g} {parity} "
                    f"R={sub.iloc[idx]['R']:g}->{sub.iloc[idx + 1]['R']:g}",
                    flush=True,
                )
                rows.append(
                    audit_segment(
                        row_a=sub.iloc[idx],
                        row_b=sub.iloc[idx + 1],
                        tube_factor=float(tube_factor),
                        quad_nodes=quad_nodes,
                        n_per_side=int(args.n_per_side),
                        dR_step=float(args.dR_step),
                        dE_step_fraction=float(args.dE_step_fraction),
                        include_center=not bool(args.no_center),
                        cfg=cfg,
                        p=p,
                    )
                )

    out_detail = pd.DataFrame(rows)
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
    print("[Adapted-flow summary]")
    print(out_summary.to_string(index=False))
    print("[Adaptive adapted-flow summary]")
    print(out_adaptive.to_string(index=False))


if __name__ == "__main__":
    main()

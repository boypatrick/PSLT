#!/usr/bin/env python3
"""
Finite-amplitude projective/Jost transfer audit for the Whittaker-DtN lift.

This is the theorem-facing follow-up to the S11 condition-number audit.  It
does not solve for new roots and does not introduce a new static-width object.
For each already-audited Whittaker-DtN root, it rewrites the Riccati flow as
the fractional-linear action of the second-order transfer matrix

    [u(R)  ]   [A B] [u(R0)  ],
    [u'(R) ] = [C D] [u'(R0) ],

so that a logarithmic derivative a=u'(R0)/u(R0) is mapped by

    T(a) = (C + D a) / (A + B a).

The finite-amplitude identity

    T(a)-T(b) = det(M) (a-b) / ((A+B a)(A+B b))

is exact.  The audit measures whether the corresponding Jost-denominator
floor on the segment between the two dyadic tail data is strong enough to
prove the local Rouche budget inherited from S8.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_projective_transfer_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_projective_transfer_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_projective_transfer_budget.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from audit_static_width_siegert_extraction import OUTDIR, u_eff
from extract_static_width_exact_dtn import (
    ExactDtnConfig,
    exterior_dtn,
    outgoing_whittaker_log_derivative,
)
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_DETAIL = OUTDIR / "static_width_whittaker_dtn_dps50_detail.csv"
DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_projective_transfer"


def parse_pairs(raw: str) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        out.append((float(left), float(right)))
    return tuple(out)


def argument_lookup(argument: pd.DataFrame) -> dict[tuple[float, float, float, str], pd.Series]:
    out: dict[tuple[float, float, float, str], pd.Series] = {}
    for _, row in argument.iterrows():
        out[(float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))] = row
    return out


def segment_min_abs(z0: complex, z1: complex) -> float:
    """Return min_{0<=t<=1} |(1-t) z0 + t z1|."""

    dz = z1 - z0
    denom = abs(dz) ** 2
    if denom <= 1.0e-300:
        return float(abs(z0))
    t_star = -float(np.real(z0 * np.conjugate(dz))) / denom
    t = min(1.0, max(0.0, t_star))
    return float(abs(z0 + t * dz))


def point_segment_distance(point: complex, z0: complex, z1: complex) -> float:
    return segment_min_abs(z0 - point, z1 - point)


def propagate_jost_solution(
    E: complex,
    D: float,
    ell: int,
    R_start: float,
    R: float,
    slope_start: complex,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[complex, complex, float, complex, float]:
    """Propagate one Jost-normalized solution with periodic real rescaling.

    The actual endpoint values are exp(log_scale) * (u_scaled, up_scaled).
    The positive real rescaling avoids the ill-conditioned full transfer
    matrix while preserving the complex phase needed by the projective floor.
    """

    def rhs(z: float, y: np.ndarray) -> list[float]:
        u = complex(float(y[0]), float(y[1]))
        up = complex(float(y[2]), float(y[3]))
        U_z = float(u_eff(np.array([z]), D=D, ell=ell, p=p)[0])
        upp = (U_z - E) * u
        return [float(up.real), float(up.imag), float(upp.real), float(upp.imag)]

    z = float(R_start)
    u = 1.0 + 0.0j
    up = slope_start
    log_scale = 0.0
    nfev = 0.0
    # Use the same local step scale as the Riccati audit, but renormalize only
    # every few solver steps.  The transfer is projective, so positive real
    # rescaling does not change slopes or pole geometry.
    chunk = max(5.0 * cfg.dtn_max_step, cfg.dtn_max_step)
    direction = -1.0 if R < R_start else 1.0
    while (z - R) * direction < -1.0e-12:
        z_next = z + direction * min(chunk, abs(R - z))
        sol = solve_ivp(
            rhs,
            t_span=(z, z_next),
            y0=np.array([u.real, u.imag, up.real, up.imag], dtype=float),
            method="DOP853",
            rtol=cfg.dtn_rtol,
            atol=cfg.dtn_atol,
            max_step=cfg.dtn_max_step,
        )
        if not sol.success:
            raise RuntimeError(f"Jost propagation failed: {sol.message}")
        u = complex(float(sol.y[0, -1]), float(sol.y[1, -1]))
        up = complex(float(sol.y[2, -1]), float(sol.y[3, -1]))
        nfev += float(sol.nfev)
        scale = max(abs(u), abs(up), 1.0e-300)
        u /= scale
        up /= scale
        log_scale += float(np.log(scale))
        z = float(z_next)
    slope_R = up / u
    return u, up, log_scale, slope_R, nfev


def transfer_matrix(
    E: complex,
    D: float,
    ell: int,
    R_start: float,
    R: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[np.ndarray, float, float]:
    """Short-range diagnostic transfer matrix from R_start down to R."""

    def rhs(z: float, y: np.ndarray) -> list[float]:
        M = np.array(
            [
                [complex(float(y[0]), float(y[1])), complex(float(y[2]), float(y[3]))],
                [complex(float(y[4]), float(y[5])), complex(float(y[6]), float(y[7]))],
            ],
            dtype=complex,
        )
        U_z = float(u_eff(np.array([z]), D=D, ell=ell, p=p)[0])
        A_z = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [U_z - E, 0.0 + 0.0j]], dtype=complex)
        dM = A_z @ M
        return [
            float(dM[0, 0].real),
            float(dM[0, 0].imag),
            float(dM[0, 1].real),
            float(dM[0, 1].imag),
            float(dM[1, 0].real),
            float(dM[1, 0].imag),
            float(dM[1, 1].real),
            float(dM[1, 1].imag),
        ]

    sol = solve_ivp(
        rhs,
        t_span=(R_start, R),
        y0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float),
        method="DOP853",
        rtol=cfg.dtn_rtol,
        atol=cfg.dtn_atol,
        max_step=cfg.dtn_max_step,
    )
    y = sol.y[:, -1]
    M = np.array(
        [
            [complex(float(y[0]), float(y[1])), complex(float(y[2]), float(y[3]))],
            [complex(float(y[4]), float(y[5])), complex(float(y[6]), float(y[7]))],
        ],
        dtype=complex,
    )
    det_m = np.linalg.det(M)
    return M, float(abs(det_m - 1.0)), float(sol.nfev)


def mobius_slope(M: np.ndarray, a: complex) -> tuple[complex, complex]:
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]
    denom = A + B * a
    return (C + D * a) / denom, denom


def audit_row_pair(
    row: pd.Series,
    arg: pd.Series,
    pair: tuple[float, float],
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    D = float(row["D"])
    R = float(row["R"])
    R_tail = float(row["R_tail"])
    parity = str(row["parity"])
    ell = int(round(float(row["ell"])))
    h = float(row["h"])
    E0 = complex(float(row["E_dtn_real"]), float(row["E_dtn_imag"]))
    R_start, R_ref = pair
    pair_label = f"{R_start:g}_to_{R_ref:g}"
    cfg_row = replace(cfg, ell=ell)

    a_start, _ = outgoing_whittaker_log_derivative(
        E=E0,
        z=R_start,
        ell=ell,
        p=p,
        dps=cfg.whittaker_dps,
    )
    b_start, _ = exterior_dtn(
        E=E0,
        R=R_start,
        R_tail=R_ref,
        D=D,
        ell=ell,
        cfg=cfg_row,
        p=p,
    )
    initial_delta = float(abs(a_start - b_start))

    u_a, up_a, log_scale_a, Ta, nfev_a = propagate_jost_solution(
        E=E0,
        D=D,
        ell=ell,
        R_start=R_start,
        R=R,
        slope_start=a_start,
        cfg=cfg_row,
        p=p,
    )
    u_b, up_b, log_scale_b, Tb, nfev_b = propagate_jost_solution(
        E=E0,
        D=D,
        ell=ell,
        R_start=R_start,
        R=R,
        slope_start=b_start,
        cfg=cfg_row,
        p=p,
    )
    transfer_delta = float(abs(Ta - Tb))

    log_product = (
        log_scale_a
        + log_scale_b
        + float(np.log(max(abs(u_a), 1.0e-300)))
        + float(np.log(max(abs(u_b), 1.0e-300)))
    )
    exact_projective_gain = float(np.exp(-log_product))
    exact_formula_delta = exact_projective_gain * initial_delta
    exact_formula_rel_error = float(
        abs(exact_formula_delta - transfer_delta) / max(transfer_delta, 1.0e-300)
    )

    common_log_scale = max(log_scale_a, log_scale_b)
    denom_a_scaled = u_a * np.exp(log_scale_a - common_log_scale)
    denom_b_scaled = u_b * np.exp(log_scale_b - common_log_scale)
    denom_floor_endpoints_scaled = float(min(abs(denom_a_scaled), abs(denom_b_scaled)))
    denom_floor_segment_scaled = segment_min_abs(denom_a_scaled, denom_b_scaled)
    denom_floor_endpoints_log = common_log_scale + float(np.log(max(denom_floor_endpoints_scaled, 1.0e-300)))
    denom_floor_segment_log = common_log_scale + float(np.log(max(denom_floor_segment_scaled, 1.0e-300)))
    denom_floor_endpoints = float(np.exp(denom_floor_endpoints_log))
    denom_floor_segment = float(np.exp(denom_floor_segment_log))
    segment_gain_bound = float(np.exp(-2.0 * denom_floor_segment_log))
    segment_delta_bound = segment_gain_bound * initial_delta

    B_scaled = (denom_a_scaled - denom_b_scaled) / (a_start - b_start)
    A_scaled = denom_a_scaled - B_scaled * a_start
    if abs(B_scaled) > 1.0e-300:
        pole_initial = -A_scaled / B_scaled
        pole_distance_segment = point_segment_distance(pole_initial, a_start, b_start)
    else:
        pole_initial = complex(np.nan, np.nan)
        pole_distance_segment = np.inf

    matrix_det_minus_one_abs = np.nan
    matrix_nfev = np.nan
    if R_start <= 800.0:
        # A raw fundamental matrix is numerically useful only on the shorter
        # intervals.  It is retained as a diagnostic, not as the theorem route.
        _, matrix_det_minus_one_abs, matrix_nfev = transfer_matrix(
            E=E0,
            D=D,
            ell=ell,
            R_start=R_start,
            R=R,
            cfg=cfg_row,
            p=p,
        )

    boundary_gap = float(arg["boundary_F_abs_min"])
    m_budget = 0.5 * h * boundary_gap

    return {
        "D": D,
        "R": R,
        "R_tail_root": R_tail,
        "parity": parity,
        "ell": float(ell),
        "h": h,
        "E0_real": float(E0.real),
        "E0_imag": float(E0.imag),
        "Gamma_dtn": float(row["Gamma_dtn"]),
        "transfer_pair": pair_label,
        "R_start": R_start,
        "R_ref": R_ref,
        "boundary_F_abs_min": boundary_gap,
        "m_rouche_budget": m_budget,
        "a_start_real": float(a_start.real),
        "a_start_imag": float(a_start.imag),
        "b_start_real": float(b_start.real),
        "b_start_imag": float(b_start.imag),
        "initial_delta_at_R_start": initial_delta,
        "transfer_delta_at_R": transfer_delta,
        "transfer_delta_over_budget": transfer_delta / max(m_budget, 1.0e-300),
        "jost_log_scale_a": float(log_scale_a),
        "jost_log_scale_b": float(log_scale_b),
        "jost_nfev_a": nfev_a,
        "jost_nfev_b": nfev_b,
        "matrix_det_minus_one_abs_short_diag": float(matrix_det_minus_one_abs),
        "matrix_nfev_short_diag": float(matrix_nfev),
        "jost_denom_a_abs": float(np.exp(log_scale_a) * abs(u_a)),
        "jost_denom_b_abs": float(np.exp(log_scale_b) * abs(u_b)),
        "jost_denom_product_abs": float(np.exp(log_product)),
        "jost_denom_floor_endpoints": denom_floor_endpoints,
        "jost_denom_floor_segment": denom_floor_segment,
        "jost_pole_initial_real": float(pole_initial.real),
        "jost_pole_initial_imag": float(pole_initial.imag),
        "jost_pole_distance_segment": float(pole_distance_segment),
        "exact_projective_gain": exact_projective_gain,
        "exact_projective_delta": exact_formula_delta,
        "exact_projective_delta_over_budget": exact_formula_delta / max(m_budget, 1.0e-300),
        "exact_projective_formula_rel_error": exact_formula_rel_error,
        "exact_projective_formula_abs_error_over_budget": abs(exact_formula_delta - transfer_delta)
        / max(m_budget, 1.0e-300),
        "segment_gain_bound": segment_gain_bound,
        "segment_delta_bound": segment_delta_bound,
        "segment_delta_bound_over_budget": segment_delta_bound / max(m_budget, 1.0e-300),
        "exact_projective_status": "passes_exact_projective_budget"
        if transfer_delta <= m_budget
        else "not_lifted_exact_projective",
        "segment_projective_status": "passes_segment_projective_budget"
        if segment_delta_bound <= m_budget
        else "not_lifted_segment_projective",
    }


def summary_row(pair: str, D: float | str, sub: pd.DataFrame) -> dict[str, float | str]:
    return {
        "transfer_pair": pair,
        "D": float(D) if isinstance(D, (float, int, np.floating)) else str(D),
        "n_rows": float(len(sub)),
        "n_exact_projective_pass": float((sub["exact_projective_status"] == "passes_exact_projective_budget").sum()),
        "n_segment_projective_pass": float(
            (sub["segment_projective_status"] == "passes_segment_projective_budget").sum()
        ),
        "transfer_delta_over_budget_min": float(sub["transfer_delta_over_budget"].min()),
        "transfer_delta_over_budget_median": float(sub["transfer_delta_over_budget"].median()),
        "transfer_delta_over_budget_max": float(sub["transfer_delta_over_budget"].max()),
        "segment_delta_bound_over_budget_min": float(sub["segment_delta_bound_over_budget"].min()),
        "segment_delta_bound_over_budget_median": float(sub["segment_delta_bound_over_budget"].median()),
        "segment_delta_bound_over_budget_max": float(sub["segment_delta_bound_over_budget"].max()),
        "exact_projective_gain_min": float(sub["exact_projective_gain"].min()),
        "exact_projective_gain_median": float(sub["exact_projective_gain"].median()),
        "exact_projective_gain_max": float(sub["exact_projective_gain"].max()),
        "segment_gain_bound_max": float(sub["segment_gain_bound"].max()),
        "initial_delta_min": float(sub["initial_delta_at_R_start"].min()),
        "initial_delta_max": float(sub["initial_delta_at_R_start"].max()),
        "jost_denom_floor_segment_min": float(sub["jost_denom_floor_segment"].min()),
        "jost_denom_floor_segment_median": float(sub["jost_denom_floor_segment"].median()),
        "jost_pole_distance_segment_min": float(sub["jost_pole_distance_segment"].min()),
        "matrix_det_minus_one_abs_short_diag_max": float(sub["matrix_det_minus_one_abs_short_diag"].max(skipna=True)),
        "exact_projective_formula_rel_error_max": float(sub["exact_projective_formula_rel_error"].max()),
        "exact_projective_formula_abs_error_over_budget_max": float(
            sub["exact_projective_formula_abs_error_over_budget"].max()
        ),
        "projective_transfer_status": "has_uniform_segment_projective_pass"
        if float((sub["segment_projective_status"] == "passes_segment_projective_budget").sum()) == len(sub)
        else "not_uniform_on_current_grid",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (pair, D), sub in detail.groupby(["transfer_pair", "D"], sort=True):
        rows.append(summary_row(pair=pair, D=D, sub=sub))
    for pair, sub in detail.groupby("transfer_pair", sort=True):
        rows.append(summary_row(pair=pair, D="global", sub=sub))
    rows.append(summary_row(pair="all_pairs", D="global", sub=detail))
    return pd.DataFrame(rows)


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    colors = {"400_to_800": "#d95f02", "800_to_1600": "#1b9e77", "1600_to_3200": "#386cb0"}

    ax = axes[0]
    for pair, sub in detail.groupby("transfer_pair", sort=True):
        ordered = sub.sort_values("transfer_delta_over_budget").reset_index(drop=True)
        x = np.arange(len(ordered))
        color = colors.get(pair, None)
        ax.scatter(x, ordered["transfer_delta_over_budget"], s=26, alpha=0.82, label=f"{pair} exact", color=color)
        ax.scatter(
            x,
            ordered["segment_delta_bound_over_budget"],
            s=18,
            alpha=0.35,
            marker="x",
            label=f"{pair} segment",
            color=color,
        )
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("rows sorted within each pair")
    ax.set_ylabel("projective dyadic delta / local m-budget")
    ax.set_title("Finite-amplitude projective budget")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    for pair, sub in detail.groupby("transfer_pair", sort=True):
        ax.scatter(
            sub["jost_denom_floor_segment"],
            sub["exact_projective_gain"],
            s=34,
            alpha=0.78,
            label=pair,
            color=colors.get(pair, None),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("segment Jost denominator floor")
    ax.set_ylabel("exact finite projective gain")
    ax.set_title("Amplification is denominator geometry")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT)
    ap.add_argument("--pairs", default="400:800,800:1600,1600:3200")
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    detail_in = pd.read_csv(args.detail)
    argument = pd.read_csv(args.argument_detail)
    arg_rows = argument_lookup(argument)
    p = PhysicalParams()
    cfg = ExactDtnConfig(tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    pairs = parse_pairs(args.pairs)

    rows: list[dict[str, float | str]] = []
    for _, row in detail_in.iterrows():
        arg = arg_rows[(float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))]
        for pair in pairs:
            rows.append(audit_row_pair(row=row, arg=arg, pair=pair, cfg=cfg, p=p))

    detail = pd.DataFrame(rows)
    summary = summarize(detail)

    detail_path = OUTDIR / f"{args.out_tag}_detail.csv"
    summary_path = OUTDIR / f"{args.out_tag}_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))

    if not args.skip_plot:
        plot_path = OUTDIR / f"{args.out_tag}_budget.png"
        plot_budget(detail, plot_path)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Local argument-principle audit for the Whittaker-DtN static-width residual.

This is the next theorem gate after the resolved-branch audit.  It does not
search for a new branch.  For each Whittaker-DtN root E0 already recorded in
the S6 detail table, it evaluates the same finite-interval residual

    F_{parity}(E; D, R, R_tail)

on a small rectangular contour around E0 and computes

    ind(F, 0) = (1 / 2 pi) Delta arg F(E).

For a holomorphic pole-free residual in the rectangle, this index is the number
of zeros inside.  The finite-difference parity residual is meromorphic, so the
script reports this as an auditable local argument index plus boundary gap and
conditioning diagnostics.  A stable promotion theorem would still need a
pole-free/condition-number argument across the R and R_tail limits.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_argument_principle_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_argument_principle_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_argument_principle_index.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR, u_eff
from extract_static_width_exact_dtn import (
    ExactDtnConfig,
    exterior_dtn,
    ratio_residual_exact_dtn,
)
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_DETAIL = OUTDIR / "static_width_whittaker_dtn_dps50_detail.csv"
DEFAULT_TAG = "static_width_whittaker_argument_principle"


def parse_floats(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def parse_strings(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def contour_rectangle(center: complex, half_re: float, half_im: float, n_per_side: int) -> np.ndarray:
    n = max(4, int(n_per_side))
    re0, im0 = center.real, center.imag
    top = re0 + np.linspace(-half_re, half_re, n, endpoint=False) + 1j * (im0 + half_im)
    right = (re0 + half_re) + 1j * (im0 + np.linspace(half_im, -half_im, n, endpoint=False))
    bottom = re0 + np.linspace(half_re, -half_re, n, endpoint=False) + 1j * (im0 - half_im)
    left = (re0 - half_re) + 1j * (im0 + np.linspace(-half_im, half_im, n, endpoint=False))
    return np.concatenate([top, right, bottom, left])


def argument_index(values: np.ndarray) -> tuple[float, int, float]:
    vals = np.asarray(values, dtype=complex)
    phases = np.unwrap(np.angle(np.concatenate([vals, vals[:1]])))
    raw = float((phases[-1] - phases[0]) / (2.0 * np.pi))
    rounded = int(round(raw))
    return raw, rounded, float(abs(raw - rounded))


def finite_row_filter(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["E_dtn_real", "E_dtn_imag", "Gamma_dtn", "residual_abs"]
    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        mask &= np.isfinite(df[col].to_numpy(dtype=float))
    return df[mask].copy()


def residual_evaluator(
    D: float,
    R: float,
    R_tail: float,
    parity: str,
    n_half: int,
    ell: int,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[callable, float]:
    h = R / n_half
    z_half = np.linspace(0.0, R, n_half + 1)
    U_half = u_eff(z_half, D=D, ell=ell, p=p)

    def eval_F(E: complex) -> complex:
        m_R, _ = exterior_dtn(E=E, R=R, R_tail=R_tail, D=D, ell=ell, cfg=cfg, p=p)
        return ratio_residual_exact_dtn(E=E, U_half=U_half, h=h, parity=parity, m_R=m_R)

    return eval_F, h


def derivative_proxy(eval_F: callable, E0: complex, step: float) -> tuple[complex, complex]:
    # Two directional complex derivatives are retained as a holomorphy sanity
    # check for the numerical backend on this small scale.
    step = max(float(step), 1.0e-8)
    d_real = (eval_F(E0 + step) - eval_F(E0 - step)) / (2.0 * step)
    d_imag = (eval_F(E0 + 1j * step) - eval_F(E0 - 1j * step)) / (2.0j * step)
    return d_real, d_imag


def audit_row(row: pd.Series, cfg: ExactDtnConfig, p: PhysicalParams, args: argparse.Namespace) -> dict[str, float | str]:
    D = float(row["D"])
    R = float(row["R"])
    R_tail = float(row["R_tail"])
    parity = str(row["parity"])
    ell = int(round(float(row.get("ell", cfg.ell))))
    n_half = int(round(float(row.get("n_half", cfg.n_half))))
    E0 = complex(float(row["E_dtn_real"]), float(row["E_dtn_imag"]))

    eval_F, h = residual_evaluator(
        D=D,
        R=R,
        R_tail=R_tail,
        parity=parity,
        n_half=n_half,
        ell=ell,
        cfg=replace(cfg, ell=ell, n_half=n_half),
        p=p,
    )
    contour = contour_rectangle(
        center=E0,
        half_re=float(args.box_re),
        half_im=float(args.box_im),
        n_per_side=int(args.n_per_side),
    )
    F_vals = np.asarray([eval_F(complex(E)) for E in contour], dtype=complex)
    index_raw, index_round, index_error = argument_index(F_vals)
    min_abs = float(np.min(np.abs(F_vals)))
    max_abs = float(np.max(np.abs(F_vals)))
    root_res = complex(eval_F(E0))
    root_res_abs = float(abs(root_res))

    d_step = min(float(args.box_re), float(args.box_im)) * float(args.derivative_step_fraction)
    d_real, d_imag = derivative_proxy(eval_F=eval_F, E0=E0, step=d_step)
    d_abs = float(0.5 * (abs(d_real) + abs(d_imag)))
    cr_rel_mismatch = float(abs(d_real - d_imag) / max(d_abs, 1.0e-300))
    newton_radius_proxy = float(root_res_abs / max(d_abs, 1.0e-300))
    boundary_root_gap_ratio = float(min_abs / max(root_res_abs, 1.0e-300))
    contour_radius_min = min(float(args.box_re), float(args.box_im))
    normalized_newton_radius = float(newton_radius_proxy / max(contour_radius_min, 1.0e-300))

    argument_zero_count_abs = abs(index_round)
    pass_local = (
        argument_zero_count_abs == 1
        and index_error <= float(args.index_error_max)
        and min_abs >= float(args.boundary_abs_min)
        and boundary_root_gap_ratio >= float(args.boundary_root_gap_ratio_min)
        and normalized_newton_radius <= float(args.newton_radius_contour_frac_max)
    )
    return {
        "D": D,
        "R": R,
        "R_tail": R_tail,
        "parity": parity,
        "ell": float(ell),
        "n_half": float(n_half),
        "h": float(h),
        "E0_real": float(E0.real),
        "E0_imag": float(E0.imag),
        "Gamma_dtn": float(row["Gamma_dtn"]),
        "root_residual_table_abs": float(row["residual_abs"]),
        "root_residual_recomputed_abs": root_res_abs,
        "box_re": float(args.box_re),
        "box_im": float(args.box_im),
        "n_contour": float(len(contour)),
        "argument_index_raw": index_raw,
        "argument_index_round": float(index_round),
        "argument_zero_count_abs": float(argument_zero_count_abs),
        "argument_index_error": index_error,
        "boundary_F_abs_min": min_abs,
        "boundary_F_abs_max": max_abs,
        "boundary_root_gap_ratio": boundary_root_gap_ratio,
        "dF_dE_abs_proxy": d_abs,
        "dF_dE_CR_rel_mismatch": cr_rel_mismatch,
        "newton_radius_proxy": newton_radius_proxy,
        "newton_radius_over_contour": normalized_newton_radius,
        "local_argument_pass": float(pass_local),
        "local_argument_status": "local_index_one_isolated_zero" if pass_local else "not_certified",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        rows.append(
            {
                "D": float(D),
                "n_rows": float(len(sub)),
                "n_local_argument_pass": float(sub["local_argument_pass"].sum()),
                "index_round_min": float(sub["argument_index_round"].min()),
                "index_round_max": float(sub["argument_index_round"].max()),
                "zero_count_abs_min": float(sub["argument_zero_count_abs"].min()),
                "zero_count_abs_max": float(sub["argument_zero_count_abs"].max()),
                "index_error_max": float(sub["argument_index_error"].max()),
                "boundary_F_abs_min": float(sub["boundary_F_abs_min"].min()),
                "boundary_root_gap_ratio_min": float(sub["boundary_root_gap_ratio"].min()),
                "dF_dE_abs_proxy_min": float(sub["dF_dE_abs_proxy"].min()),
                "dF_dE_CR_rel_mismatch_max": float(sub["dF_dE_CR_rel_mismatch"].max()),
                "newton_radius_proxy_max": float(sub["newton_radius_proxy"].max()),
                "newton_radius_over_contour_max": float(sub["newton_radius_over_contour"].max()),
                "local_argument_status": "all_local_index_one"
                if float(sub["local_argument_pass"].sum()) == len(sub)
                else "not_all_certified",
            }
        )
    rows.append(
        {
            "D": "global",
            "n_rows": float(len(detail)),
            "n_local_argument_pass": float(detail["local_argument_pass"].sum()),
            "index_round_min": float(detail["argument_index_round"].min()),
            "index_round_max": float(detail["argument_index_round"].max()),
            "zero_count_abs_min": float(detail["argument_zero_count_abs"].min()),
            "zero_count_abs_max": float(detail["argument_zero_count_abs"].max()),
            "index_error_max": float(detail["argument_index_error"].max()),
            "boundary_F_abs_min": float(detail["boundary_F_abs_min"].min()),
            "boundary_root_gap_ratio_min": float(detail["boundary_root_gap_ratio"].min()),
            "dF_dE_abs_proxy_min": float(detail["dF_dE_abs_proxy"].min()),
            "dF_dE_CR_rel_mismatch_max": float(detail["dF_dE_CR_rel_mismatch"].max()),
            "newton_radius_proxy_max": float(detail["newton_radius_proxy"].max()),
            "newton_radius_over_contour_max": float(detail["newton_radius_over_contour"].max()),
            "local_argument_status": "all_local_index_one"
            if float(detail["local_argument_pass"].sum()) == len(detail)
            else "not_all_certified",
        }
    )
    return pd.DataFrame(rows)


def plot_index(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    for D, sub in detail.groupby("D", sort=True):
        ax.scatter(sub["E0_real"], sub["E0_imag"], s=48, label=f"D={D:g}", alpha=0.82)
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xlabel(r"$\Re E_0$")
    ax.set_ylabel(r"$\Im E_0$")
    ax.set_title("Whittaker-DtN roots audited by local argument index")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    jitter = np.linspace(-0.08, 0.08, len(detail))
    x = detail["argument_index_round"].to_numpy(dtype=float) + jitter
    color = np.where(detail["local_argument_pass"].to_numpy(dtype=float) > 0.5, "#1d6fa5", "#c45a2a")
    ax.scatter(x, detail["boundary_F_abs_min"], c=color, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("rounded argument index")
    ax.set_ylabel(r"$\min_{\partial Q}|F|$")
    ax.set_title("Boundary gap")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local argument-principle audit for Whittaker-DtN roots.")
    ap.add_argument("--detail-csv", default=str(DEFAULT_DETAIL))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="", help="Optional comma-separated D filter.")
    ap.add_argument("--R-values", default="", help="Optional comma-separated R filter.")
    ap.add_argument("--R-tail-values", default="", help="Optional comma-separated R_tail filter.")
    ap.add_argument("--parities", default="", help="Optional comma-separated parity filter.")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional row cap for smoke tests.")
    ap.add_argument("--box-re", type=float, default=1.0e-4)
    ap.add_argument("--box-im", type=float, default=1.0e-4)
    ap.add_argument("--n-per-side", type=int, default=16)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--derivative-step-fraction", type=float, default=0.10)
    ap.add_argument("--index-error-max", type=float, default=5.0e-2)
    ap.add_argument("--boundary-abs-min", type=float, default=1.0e-6)
    ap.add_argument("--boundary-root-gap-ratio-min", type=float, default=25.0)
    ap.add_argument("--newton-radius-contour-frac-max", type=float, default=0.25)
    ap.add_argument("--skip-plot", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    detail_csv = Path(args.detail_csv)
    if not detail_csv.exists():
        raise FileNotFoundError(f"Missing Whittaker detail CSV: {detail_csv}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    detail = finite_row_filter(pd.read_csv(detail_csv))
    Ds = parse_floats(args.Ds)
    Rs = parse_floats(args.R_values)
    Rtails = parse_floats(args.R_tail_values)
    parities = parse_strings(args.parities)
    if Ds is not None:
        detail = detail[detail["D"].isin(Ds)]
    if Rs is not None:
        detail = detail[detail["R"].isin(Rs)]
    if Rtails is not None:
        detail = detail[detail["R_tail"].isin(Rtails)]
    if parities is not None:
        detail = detail[detail["parity"].isin(parities)]
    detail = detail.sort_values(["D", "R_tail", "R", "parity"]).reset_index(drop=True)
    if args.max_rows and args.max_rows > 0:
        detail = detail.head(int(args.max_rows)).copy()
    if len(detail) == 0:
        raise ValueError("No rows selected for argument-principle audit.")

    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()
    rows = []
    for idx, row in detail.iterrows():
        print(
            f"[{idx + 1}/{len(detail)}] D={row['D']:g} R={row['R']:g} "
            f"R_tail={row['R_tail']:g} parity={row['parity']}",
            flush=True,
        )
        rows.append(audit_row(row=row, cfg=cfg, p=p, args=args))
    out_detail = pd.DataFrame(rows)
    out_summary = summarize(out_detail)

    detail_out = OUTDIR / f"{args.out_tag}_detail.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    out_detail.to_csv(detail_out, index=False)
    out_summary.to_csv(summary_out, index=False)

    plot_out = OUTDIR / f"{args.out_tag}_index.png"
    if not args.skip_plot:
        plot_index(out_detail, plot_out)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    if not args.skip_plot:
        print(f"Wrote {plot_out}")
    cols = [
        "D",
        "n_rows",
        "n_local_argument_pass",
        "index_round_min",
        "index_round_max",
        "zero_count_abs_min",
        "zero_count_abs_max",
        "boundary_F_abs_min",
        "boundary_root_gap_ratio_min",
        "newton_radius_over_contour_max",
        "local_argument_status",
    ]
    print("[Whittaker-DtN local argument-principle summary]")
    print(out_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

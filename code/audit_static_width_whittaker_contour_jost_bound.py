#!/usr/bin/env python3
"""
Contour-uniform Jost-product / projective pole-exclusion audit.

This is the local-box lift of S13.  It keeps the same Whittaker-DtN roots and
the same parent dyadic pair, and tests the existing S8 argument boxes.  No new
root family, backend, or promotion object is introduced.

For each local argument box Q around an audited root, define on the boundary

    F(E) = (a(E)-b(E)) / (B_m u_a(R;E) u_b(R;E)),

where a is the outgoing Whittaker slope at R0, b is the slope obtained by
propagating the outgoing data from 2R0 to R0, and B_m is the fixed S8 local
Rouche m-budget.  If |F|<1 on dQ and both endpoint Jost factors have zero
winding around zero, then the maximum-modulus principle and the argument
principle promote the S13 endpoint bound to the whole local box.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_contour_jost_bound_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_contour_jost_bound_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_contour_jost_bound_budget.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_projective_transfer import propagate_jost_solution
from extract_static_width_exact_dtn import (
    ExactDtnConfig,
    exterior_dtn,
    outgoing_whittaker_log_derivative,
)
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_DETAIL = OUTDIR / "static_width_whittaker_dtn_dps50_detail.csv"
DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_contour_jost_bound"


def parse_pair(raw: str) -> tuple[float, float]:
    left, right = raw.split(":")
    return float(left), float(right)


def parse_float_list(raw: str) -> set[float]:
    if not raw.strip():
        return set()
    return {float(part.strip()) for part in raw.split(",") if part.strip()}


def parse_string_list(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def argument_lookup(argument: pd.DataFrame) -> dict[tuple[float, float, float, str], pd.Series]:
    out: dict[tuple[float, float, float, str], pd.Series] = {}
    for _, row in argument.iterrows():
        out[(float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))] = row
    return out


def rectangle_contour(center: complex, half_re: float, half_im: float, n_total: int) -> np.ndarray:
    n_side = max(2, int(n_total) // 4)
    xs = np.linspace(-half_re, half_re, n_side, endpoint=False)
    ys = np.linspace(-half_im, half_im, n_side, endpoint=False)
    pts: list[complex] = []
    pts.extend(center + x - 1j * half_im for x in xs)
    pts.extend(center + half_re + 1j * y for y in ys)
    pts.extend(center - x + 1j * half_im for x in xs)
    pts.extend(center - half_re - 1j * y for y in ys)
    return np.asarray(pts, dtype=complex)


def winding_number(values: np.ndarray) -> tuple[float, int]:
    closed = np.r_[values, values[0]]
    angles = np.unwrap(np.angle(closed))
    raw = float((angles[-1] - angles[0]) / (2.0 * np.pi))
    return raw, int(round(raw))


def eval_contour_point(
    E: complex,
    D: float,
    R: float,
    ell: int,
    R_start: float,
    R_ref: float,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> dict[str, complex | float]:
    a_start, _ = outgoing_whittaker_log_derivative(
        E=E,
        z=R_start,
        ell=ell,
        p=p,
        dps=cfg.whittaker_dps,
    )
    b_start, _ = exterior_dtn(
        E=E,
        R=R_start,
        R_tail=R_ref,
        D=D,
        ell=ell,
        cfg=cfg,
        p=p,
    )
    u_a, _up_a, log_scale_a, T_a, nfev_a = propagate_jost_solution(
        E=E,
        D=D,
        ell=ell,
        R_start=R_start,
        R=R,
        slope_start=a_start,
        cfg=cfg,
        p=p,
    )
    u_b, _up_b, log_scale_b, T_b, nfev_b = propagate_jost_solution(
        E=E,
        D=D,
        ell=ell,
        R_start=R_start,
        R=R,
        slope_start=b_start,
        cfg=cfg,
        p=p,
    )
    jost_a = np.exp(log_scale_a) * u_a
    jost_b = np.exp(log_scale_b) * u_b
    return {
        "a_start": a_start,
        "b_start": b_start,
        "initial_delta": float(abs(a_start - b_start)),
        "jost_a": jost_a,
        "jost_b": jost_b,
        "jost_product": float(abs(jost_a * jost_b)),
        "transfer_delta": float(abs(T_a - T_b)),
        "nfev": float(nfev_a + nfev_b),
    }


def audit_row(
    row: pd.Series,
    arg: pd.Series,
    pair: tuple[float, float],
    n_contour: int,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    D = float(row["D"])
    R = float(row["R"])
    R_tail = float(row["R_tail"])
    parity = str(row["parity"])
    ell = int(round(float(row["ell"])))
    E0 = complex(float(row["E_dtn_real"]), float(row["E_dtn_imag"]))
    R_start, R_ref = pair
    cfg_row = replace(cfg, ell=ell)
    h = float(row["h"])
    B_m = 0.5 * h * float(arg["boundary_F_abs_min"])
    contour = rectangle_contour(
        center=E0,
        half_re=float(arg["box_re"]),
        half_im=float(arg["box_im"]),
        n_total=n_contour,
    )

    values: list[dict[str, complex | float]] = []
    for E in contour:
        values.append(
            eval_contour_point(
                E=E,
                D=D,
                R=R,
                ell=ell,
                R_start=R_start,
                R_ref=R_ref,
                cfg=cfg_row,
                p=p,
            )
        )

    jost_a = np.asarray([complex(v["jost_a"]) for v in values], dtype=complex)
    jost_b = np.asarray([complex(v["jost_b"]) for v in values], dtype=complex)
    initial_delta = np.asarray([float(v["initial_delta"]) for v in values], dtype=float)
    product = np.asarray([float(v["jost_product"]) for v in values], dtype=float)
    transfer_delta = np.asarray([float(v["transfer_delta"]) for v in values], dtype=float)
    nfev = np.asarray([float(v["nfev"]) for v in values], dtype=float)

    required_product = initial_delta / max(B_m, 1.0e-300)
    ratio_direct = initial_delta / np.maximum(B_m * product, 1.0e-300)
    ratio_transfer = transfer_delta / max(B_m, 1.0e-300)
    product_margin = product / np.maximum(required_product, 1.0e-300)
    winding_a_raw, winding_a_round = winding_number(jost_a)
    winding_b_raw, winding_b_round = winding_number(jost_b)

    pass_bound = bool(np.max(ratio_direct) < 1.0 and winding_a_round == 0 and winding_b_round == 0)
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
        "transfer_pair": f"{R_start:g}_to_{R_ref:g}",
        "R_start": float(R_start),
        "R_ref": float(R_ref),
        "box_re": float(arg["box_re"]),
        "box_im": float(arg["box_im"]),
        "n_contour": float(len(contour)),
        "m_rouche_budget": B_m,
        "boundary_required_product_min": float(np.min(required_product)),
        "boundary_required_product_median": float(np.median(required_product)),
        "boundary_required_product_max": float(np.max(required_product)),
        "boundary_jost_product_min": float(np.min(product)),
        "boundary_jost_product_median": float(np.median(product)),
        "boundary_jost_product_max": float(np.max(product)),
        "boundary_product_margin_min": float(np.min(product_margin)),
        "boundary_product_margin_median": float(np.median(product_margin)),
        "boundary_ratio_direct_max": float(np.max(ratio_direct)),
        "boundary_ratio_transfer_max": float(np.max(ratio_transfer)),
        "boundary_ratio_direct_median": float(np.median(ratio_direct)),
        "boundary_ratio_transfer_median": float(np.median(ratio_transfer)),
        "jost_a_boundary_min": float(np.min(np.abs(jost_a))),
        "jost_b_boundary_min": float(np.min(np.abs(jost_b))),
        "jost_a_winding_raw": winding_a_raw,
        "jost_b_winding_raw": winding_b_raw,
        "jost_a_winding_round": float(winding_a_round),
        "jost_b_winding_round": float(winding_b_round),
        "nfev_total": float(np.sum(nfev)),
        "contour_jost_status": "passes_contour_jost_box"
        if pass_bound
        else "not_lifted_contour_jost_box",
    }


def summary_row(label: str, sub: pd.DataFrame) -> dict[str, float | str]:
    passes = sub["contour_jost_status"] == "passes_contour_jost_box"
    return {
        "group": label,
        "n_rows": float(len(sub)),
        "n_contour_jost_pass": float(passes.sum()),
        "boundary_ratio_direct_max": float(sub["boundary_ratio_direct_max"].max()),
        "boundary_ratio_transfer_max": float(sub["boundary_ratio_transfer_max"].max()),
        "boundary_product_margin_min": float(sub["boundary_product_margin_min"].min()),
        "boundary_product_margin_median": float(sub["boundary_product_margin_median"].median()),
        "boundary_jost_product_min": float(sub["boundary_jost_product_min"].min()),
        "boundary_required_product_max": float(sub["boundary_required_product_max"].max()),
        "jost_a_boundary_min": float(sub["jost_a_boundary_min"].min()),
        "jost_b_boundary_min": float(sub["jost_b_boundary_min"].min()),
        "winding_abs_max": float(
            max(sub["jost_a_winding_round"].abs().max(), sub["jost_b_winding_round"].abs().max())
        ),
        "contour_jost_theorem_status": "has_uniform_contour_jost_box_bound"
        if float(passes.sum()) == len(sub)
        else "not_uniform_on_current_grid",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        rows.append(summary_row(f"D={D:g}", sub))
    rows.append(summary_row("global", detail))
    return pd.DataFrame(rows)


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.1))
    ordered = detail.sort_values("boundary_product_margin_min").reset_index(drop=True)
    x = np.arange(len(ordered))

    ax = axes[0]
    ax.scatter(x, ordered["boundary_product_margin_min"], s=28, alpha=0.8)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("rows sorted by contour margin")
    ax.set_ylabel("min boundary Jost product / max required product")
    ax.set_title("Contour-uniform Jost-product margin")

    ax = axes[1]
    colors = {6.0: "#d95f02", 12.0: "#1b9e77", 18.0: "#386cb0"}
    for D, sub in detail.groupby("D", sort=True):
        ax.scatter(
            sub["R"],
            sub["boundary_ratio_direct_max"],
            s=36,
            alpha=0.8,
            label=f"D={D:g}",
            color=colors.get(float(D), None),
        )
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("interface R")
    ax.set_ylabel("max boundary |F(E)|")
    ax.set_title("Maximum-modulus gate")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT)
    ap.add_argument("--pair", default="1600:3200")
    ap.add_argument("--n-contour", type=int, default=32)
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--d-values", default="", help="Comma-separated D filter, e.g. 18")
    ap.add_argument("--r-values", default="", help="Comma-separated R filter, e.g. 120")
    ap.add_argument("--parities", default="", help="Comma-separated parity filter, e.g. even,odd")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional row cap after filters")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    detail_in = pd.read_csv(args.detail)
    d_values = parse_float_list(args.d_values)
    r_values = parse_float_list(args.r_values)
    parities = parse_string_list(args.parities)
    if d_values:
        detail_in = detail_in[detail_in["D"].astype(float).isin(d_values)]
    if r_values:
        detail_in = detail_in[detail_in["R"].astype(float).isin(r_values)]
    if parities:
        detail_in = detail_in[detail_in["parity"].astype(str).isin(parities)]
    if int(args.max_rows) > 0:
        detail_in = detail_in.head(int(args.max_rows))
    detail_in = detail_in.reset_index(drop=True)
    if detail_in.empty:
        raise ValueError("No rows remain after applying filters.")

    argument = pd.read_csv(args.argument_detail)
    arg_rows = argument_lookup(argument)
    p = PhysicalParams()
    cfg = ExactDtnConfig(tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    pair = parse_pair(args.pair)

    rows: list[dict[str, float | str]] = []
    for idx, row in detail_in.iterrows():
        if args.progress:
            print(
                f"[{idx + 1}/{len(detail_in)}] D={float(row['D']):g} "
                f"R={float(row['R']):g} parity={row['parity']} "
                f"n_contour={int(args.n_contour)}",
                flush=True,
            )
        key = (float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))
        if key not in arg_rows:
            raise KeyError(f"Missing argument-principle row for {key}")
        rows.append(
            audit_row(
                row=row,
                arg=arg_rows[key],
                pair=pair,
                n_contour=int(args.n_contour),
                cfg=cfg,
                p=p,
            )
        )

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

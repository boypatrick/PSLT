#!/usr/bin/env python3
"""
Large-tube Rouché audit for the static-width R-homotopy gate.

This is the S18 theorem-facing diagnostic after the S17 finite-difference
R-flow sizing result.  It does not solve new roots, does not change the
Whittaker-DtN backend, and does not introduce a new resonance family.  It
asks the direct parent-side question:

    Can the already isolated roots at R_j and R_{j+1} be connected by a
    larger common contour on which F_{R_j} and F_{R_{j+1}} are Rouché-close?

For each fixed (D, R_tail, parity) branch label, the script builds:

  1. segment tubes around each adjacent pair R_j -> R_{j+1};
  2. optional full-path tubes around all R in {60,80,100,120}.

On each tube boundary it evaluates the same Whittaker-DtN residual used in
the S8 argument-principle audit and records

    max |F_{R_b} - F_{R_a}| / min |F_{R_a}|

plus the direct argument index of both endpoint residuals on that same
large contour.  Passing this finite audit would not by itself prove the
continuum-in-R theorem, but it would supply the correct parent-side Rouché
certificate to be lifted analytically.  Failing it tells us the next theorem
must be an implicit-flow derivative bound or an even more adapted contour,
not another local root solve.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_large_tube_rouche_segment_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_large_tube_rouche_path_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_large_tube_rouche_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_large_tube_rouche_adaptive_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_large_tube_rouche_budget.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR
from audit_static_width_whittaker_argument_principle import (
    argument_index,
    contour_rectangle,
    residual_evaluator,
)
from extract_static_width_exact_dtn import ExactDtnConfig
from extract_superrad_prefactor_1d import PhysicalParams


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_TAG = "static_width_whittaker_large_tube_rouche"


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


def make_residual(row: pd.Series, cfg: ExactDtnConfig, p: PhysicalParams) -> callable:
    ell = int(round(float(row.get("ell", cfg.ell))))
    n_half = int(round(float(row.get("n_half", cfg.n_half))))
    eval_f, _ = residual_evaluator(
        D=float(row["D"]),
        R=float(row["R"]),
        R_tail=float(row["R_tail"]),
        parity=str(row["parity"]),
        n_half=n_half,
        ell=ell,
        cfg=replace(cfg, ell=ell, n_half=n_half),
        p=p,
    )
    return eval_f


def tube_from_points(points: list[complex], box_re: float, box_im: float, pad_factor: float) -> tuple[complex, float, float]:
    re_vals = np.asarray([z.real for z in points], dtype=float)
    im_vals = np.asarray([z.imag for z in points], dtype=float)
    center = complex(0.5 * (float(re_vals.min()) + float(re_vals.max())), 0.5 * (float(im_vals.min()) + float(im_vals.max())))
    half_re = 0.5 * float(re_vals.max() - re_vals.min()) + float(pad_factor) * box_re
    half_im = 0.5 * float(im_vals.max() - im_vals.min()) + float(pad_factor) * box_im
    return center, half_re, half_im


def eval_on_contour(eval_f: callable, contour: np.ndarray) -> np.ndarray:
    return np.asarray([eval_f(complex(z)) for z in contour], dtype=complex)


def contour_stats(values: np.ndarray) -> dict[str, float]:
    idx_raw, idx_round, idx_err = argument_index(values)
    abs_vals = np.abs(values)
    return {
        "argument_index_raw": float(idx_raw),
        "argument_index_round": float(idx_round),
        "argument_index_error": float(idx_err),
        "argument_zero_count_abs": float(abs(idx_round)),
        "boundary_F_abs_min": float(np.min(abs_vals)),
        "boundary_F_abs_max": float(np.max(abs_vals)),
    }


def pair_audit(
    row_a: pd.Series,
    row_b: pd.Series,
    pad_factor: float,
    n_per_side: int,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
    contour_kind: str,
) -> dict[str, float | str]:
    e_a = root_complex(row_a)
    e_b = root_complex(row_b)
    box_re = min(float(row_a["box_re"]), float(row_b["box_re"]))
    box_im = min(float(row_a["box_im"]), float(row_b["box_im"]))
    center, half_re, half_im = tube_from_points([e_a, e_b], box_re=box_re, box_im=box_im, pad_factor=pad_factor)
    contour = contour_rectangle(center=center, half_re=half_re, half_im=half_im, n_per_side=n_per_side)

    eval_a = make_residual(row=row_a, cfg=cfg, p=p)
    eval_b = make_residual(row=row_b, cfg=cfg, p=p)
    vals_a = eval_on_contour(eval_a, contour)
    vals_b = eval_on_contour(eval_b, contour)
    delta = vals_b - vals_a

    stats_a = contour_stats(vals_a)
    stats_b = contour_stats(vals_b)
    delta_max = float(np.max(np.abs(delta)))
    ratio_a = float(delta_max / max(stats_a["boundary_F_abs_min"], 1.0e-300))
    ratio_b = float(delta_max / max(stats_b["boundary_F_abs_min"], 1.0e-300))
    symmetric_ratio = max(ratio_a, ratio_b)
    direct_index_ok = (
        abs(stats_a["argument_index_round"]) == 1.0
        and abs(stats_b["argument_index_round"]) == 1.0
        and stats_a["argument_index_error"] <= 5.0e-2
        and stats_b["argument_index_error"] <= 5.0e-2
    )
    rouche_ok = ratio_a < 1.0
    symmetric_rouche_ok = symmetric_ratio < 1.0
    status = "large_tube_rouche_pass" if (direct_index_ok and rouche_ok) else "large_tube_rouche_not_closed"
    if direct_index_ok and not rouche_ok:
        status = "large_tube_index_ok_but_rouche_ratio_large"
    elif not direct_index_ok and rouche_ok:
        status = "rouche_ratio_ok_but_index_not_one"

    return {
        "tube_kind": contour_kind,
        "D": float(row_a["D"]),
        "R_tail": float(row_a["R_tail"]),
        "parity": str(row_a["parity"]),
        "R0": float(row_a["R"]),
        "R1": float(row_b["R"]),
        "pad_factor": float(pad_factor),
        "n_contour": float(len(contour)),
        "center_real": float(center.real),
        "center_imag": float(center.imag),
        "half_re": float(half_re),
        "half_im": float(half_im),
        "half_re_over_box": float(half_re / max(box_re, 1.0e-300)),
        "half_im_over_box": float(half_im / max(box_im, 1.0e-300)),
        "root_gap_abs": float(abs(e_b - e_a)),
        "delta_F_abs_max": delta_max,
        "rouche_ratio_from_R0": ratio_a,
        "rouche_ratio_from_R1": ratio_b,
        "rouche_ratio_symmetric": symmetric_ratio,
        "R0_argument_index_round": stats_a["argument_index_round"],
        "R1_argument_index_round": stats_b["argument_index_round"],
        "R0_argument_index_error": stats_a["argument_index_error"],
        "R1_argument_index_error": stats_b["argument_index_error"],
        "R0_zero_count_abs": stats_a["argument_zero_count_abs"],
        "R1_zero_count_abs": stats_b["argument_zero_count_abs"],
        "R0_boundary_F_abs_min": stats_a["boundary_F_abs_min"],
        "R1_boundary_F_abs_min": stats_b["boundary_F_abs_min"],
        "direct_endpoint_index_ok": float(direct_index_ok),
        "rouche_pass_from_R0": float(rouche_ok),
        "symmetric_rouche_pass": float(symmetric_rouche_ok),
        "large_tube_status": status,
    }


def full_path_audit(
    sub: pd.DataFrame,
    pad_factor: float,
    n_per_side: int,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> list[dict[str, float | str]]:
    sub = sub.sort_values("R").reset_index(drop=True)
    roots = [root_complex(row) for _, row in sub.iterrows()]
    box_re = float(sub["box_re"].min())
    box_im = float(sub["box_im"].min())
    center, half_re, half_im = tube_from_points(roots, box_re=box_re, box_im=box_im, pad_factor=pad_factor)
    contour = contour_rectangle(center=center, half_re=half_re, half_im=half_im, n_per_side=n_per_side)

    evals = [make_residual(row=row, cfg=cfg, p=p) for _, row in sub.iterrows()]
    values = [eval_on_contour(eval_f, contour) for eval_f in evals]
    stats = [contour_stats(vals) for vals in values]
    rows: list[dict[str, float | str]] = []
    for idx in range(len(sub) - 1):
        row_a = sub.iloc[idx]
        row_b = sub.iloc[idx + 1]
        vals_a = values[idx]
        vals_b = values[idx + 1]
        delta_max = float(np.max(np.abs(vals_b - vals_a)))
        ratio_a = float(delta_max / max(stats[idx]["boundary_F_abs_min"], 1.0e-300))
        ratio_b = float(delta_max / max(stats[idx + 1]["boundary_F_abs_min"], 1.0e-300))
        direct_index_ok = (
            abs(stats[idx]["argument_index_round"]) == 1.0
            and abs(stats[idx + 1]["argument_index_round"]) == 1.0
            and stats[idx]["argument_index_error"] <= 5.0e-2
            and stats[idx + 1]["argument_index_error"] <= 5.0e-2
        )
        status = "full_path_large_tube_rouche_pass" if (direct_index_ok and ratio_a < 1.0) else "full_path_large_tube_not_closed"
        if direct_index_ok and ratio_a >= 1.0:
            status = "full_path_index_ok_but_rouche_ratio_large"
        rows.append(
            {
                "tube_kind": "full_path",
                "D": float(row_a["D"]),
                "R_tail": float(row_a["R_tail"]),
                "parity": str(row_a["parity"]),
                "R0": float(row_a["R"]),
                "R1": float(row_b["R"]),
                "pad_factor": float(pad_factor),
                "n_contour": float(len(contour)),
                "center_real": float(center.real),
                "center_imag": float(center.imag),
                "half_re": float(half_re),
                "half_im": float(half_im),
                "half_re_over_box": float(half_re / max(box_re, 1.0e-300)),
                "half_im_over_box": float(half_im / max(box_im, 1.0e-300)),
                "root_gap_abs": float(abs(root_complex(row_b) - root_complex(row_a))),
                "delta_F_abs_max": delta_max,
                "rouche_ratio_from_R0": ratio_a,
                "rouche_ratio_from_R1": ratio_b,
                "rouche_ratio_symmetric": max(ratio_a, ratio_b),
                "R0_argument_index_round": stats[idx]["argument_index_round"],
                "R1_argument_index_round": stats[idx + 1]["argument_index_round"],
                "R0_argument_index_error": stats[idx]["argument_index_error"],
                "R1_argument_index_error": stats[idx + 1]["argument_index_error"],
                "R0_zero_count_abs": stats[idx]["argument_zero_count_abs"],
                "R1_zero_count_abs": stats[idx + 1]["argument_zero_count_abs"],
                "R0_boundary_F_abs_min": stats[idx]["boundary_F_abs_min"],
                "R1_boundary_F_abs_min": stats[idx + 1]["boundary_F_abs_min"],
                "direct_endpoint_index_ok": float(direct_index_ok),
                "rouche_pass_from_R0": float(ratio_a < 1.0),
                "symmetric_rouche_pass": float(max(ratio_a, ratio_b) < 1.0),
                "large_tube_status": status,
            }
        )
    return rows


def summarize(segment: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for kind, detail in (("segment", segment), ("full_path", path)):
        if len(detail) == 0:
            continue
        for pad, sub in detail.groupby("pad_factor", sort=True):
            rows.append(summary_row(kind=kind, pad_factor=float(pad), sub=sub))
    return pd.DataFrame(rows)


def adaptive_summarize(segment: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for kind, detail in (("segment", segment), ("full_path", path)):
        if len(detail) == 0:
            continue
        rows.append(adaptive_summary_row(kind=kind, detail=detail))
    return pd.DataFrame(rows)


def adaptive_summary_row(kind: str, detail: pd.DataFrame) -> dict[str, float | str]:
    keys = ["D", "R_tail", "parity", "R0", "R1"]
    records: list[dict[str, float | str]] = []
    for key, sub in detail.groupby(keys, sort=True):
        ok_index = sub[sub["direct_endpoint_index_ok"] > 0.5]
        ok_rouche = sub[sub["rouche_pass_from_R0"] > 0.5]
        ok_both = sub[(sub["direct_endpoint_index_ok"] > 0.5) & (sub["rouche_pass_from_R0"] > 0.5)]
        best_any = sub.loc[sub["rouche_ratio_from_R0"].idxmin()]
        best_index = ok_index.loc[ok_index["rouche_ratio_from_R0"].idxmin()] if len(ok_index) else best_any
        records.append(
            {
                **dict(zip(keys, key)),
                "has_index": float(len(ok_index) > 0),
                "has_rouche": float(len(ok_rouche) > 0),
                "has_both": float(len(ok_both) > 0),
                "best_any_ratio": float(best_any["rouche_ratio_from_R0"]),
                "best_any_pad": float(best_any["pad_factor"]),
                "best_any_index_ok": float(best_any["direct_endpoint_index_ok"]),
                "best_index_compatible_ratio": float(best_index["rouche_ratio_from_R0"]),
                "best_index_compatible_pad": float(best_index["pad_factor"]),
                "best_index_compatible_boundary_min": float(
                    min(best_index["R0_boundary_F_abs_min"], best_index["R1_boundary_F_abs_min"])
                ),
            }
        )
    best = pd.DataFrame(records)
    n = len(best)
    n_index = int((best["has_index"] > 0.5).sum())
    n_rouche = int((best["has_rouche"] > 0.5).sum())
    n_both = int((best["has_both"] > 0.5).sum())
    status = (
        "adaptive_large_tube_rouche_closed_on_audited_grid"
        if n_both == n
        else "adaptive_large_tube_rouche_not_closed"
    )
    return {
        "tube_kind": kind,
        "n_segments": float(n),
        "n_segments_with_some_endpoint_index_ok": float(n_index),
        "n_segments_with_some_rouche_pass": float(n_rouche),
        "n_segments_with_some_index_and_rouche_pass": float(n_both),
        "max_best_any_rouche_ratio": float(best["best_any_ratio"].max()),
        "median_best_any_rouche_ratio": float(best["best_any_ratio"].median()),
        "max_best_index_compatible_rouche_ratio": float(best["best_index_compatible_ratio"].max()),
        "median_best_index_compatible_rouche_ratio": float(best["best_index_compatible_ratio"].median()),
        "min_best_index_compatible_boundary": float(best["best_index_compatible_boundary_min"].min()),
        "large_tube_status": status,
    }


def summary_row(kind: str, pad_factor: float, sub: pd.DataFrame) -> dict[str, float | str]:
    n = len(sub)
    n_index = int((sub["direct_endpoint_index_ok"] > 0.5).sum())
    n_rouche = int((sub["rouche_pass_from_R0"] > 0.5).sum())
    n_sym = int((sub["symmetric_rouche_pass"] > 0.5).sum())
    status = "large_tube_rouche_closed_on_audited_grid" if (n_index == n and n_rouche == n) else "large_tube_rouche_not_closed"
    if n_index == n and n_rouche < n:
        status = "index_count_ok_but_rouche_ratio_large"
    return {
        "tube_kind": kind,
        "pad_factor": float(pad_factor),
        "n_segments": float(n),
        "n_endpoint_index_ok": float(n_index),
        "n_rouche_pass_from_R0": float(n_rouche),
        "n_symmetric_rouche_pass": float(n_sym),
        "max_rouche_ratio_from_R0": float(sub["rouche_ratio_from_R0"].max()),
        "median_rouche_ratio_from_R0": float(sub["rouche_ratio_from_R0"].median()),
        "min_rouche_ratio_from_R0": float(sub["rouche_ratio_from_R0"].min()),
        "max_symmetric_rouche_ratio": float(sub["rouche_ratio_symmetric"].max()),
        "min_boundary_F_abs": float(min(sub["R0_boundary_F_abs_min"].min(), sub["R1_boundary_F_abs_min"].min())),
        "max_half_re_over_box": float(sub["half_re_over_box"].max()),
        "max_half_im_over_box": float(sub["half_im_over_box"].max()),
        "large_tube_status": status,
    }


def plot_budget(summary: pd.DataFrame, out_path: Path) -> None:
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/pslt_matplotlib_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/pslt_xdg_cache"))
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for kind, sub in summary.groupby("tube_kind", sort=True):
        ax.plot(
            sub["pad_factor"],
            sub["max_rouche_ratio_from_R0"],
            marker="o",
            lw=1.8,
            label=f"{kind}: max Rouché ratio",
        )
        ax.plot(
            sub["pad_factor"],
            sub["median_rouche_ratio_from_R0"],
            marker=".",
            lw=1.2,
            ls="--",
            label=f"{kind}: median",
        )
    ax.axhline(1.0, color="black", lw=1.0, ls=":", label="Rouché threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tube padding in S8 box half-widths")
    ax.set_ylabel(r"$\max_{\partial\Omega}|F_{R_1}-F_{R_0}|/\min_{\partial\Omega}|F_{R_0}|$")
    ax.set_title("Large-tube Rouché budget")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Large-tube Rouché audit for Whittaker-DtN R-homotopy.")
    ap.add_argument("--argument-csv", default=str(DEFAULT_ARGUMENT))
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--Ds", default="", help="Optional comma-separated D filter.")
    ap.add_argument("--R-tail-values", default="", help="Optional comma-separated R_tail filter.")
    ap.add_argument("--parities", default="", help="Optional comma-separated parity filter.")
    ap.add_argument("--pad-factors", default="2,4,8,16", help="Padding in units of S8 box half-width.")
    ap.add_argument("--n-per-side", type=int, default=8)
    ap.add_argument("--whittaker-dps", type=int, default=50)
    ap.add_argument("--skip-full-path", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
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
        raise ValueError("No rows selected for large-tube Rouché audit.")

    pad_factors = parse_floats(args.pad_factors)
    if pad_factors is None:
        raise ValueError("--pad-factors must contain at least one value.")

    cfg = replace(ExactDtnConfig(), tail_mode="whittaker", whittaker_dps=int(args.whittaker_dps))
    p = PhysicalParams()
    segment_rows: list[dict[str, float | str]] = []
    path_rows: list[dict[str, float | str]] = []
    groups = list(detail.groupby(["D", "R_tail", "parity"], sort=True))
    for pad_factor in pad_factors:
        print(f"[pad={pad_factor:g}] auditing {len(groups)} fixed-(D,R_tail,parity) paths", flush=True)
        for (D, R_tail, parity), sub in groups:
            sub = sub.sort_values("R").reset_index(drop=True)
            if len(sub) < 2:
                continue
            for idx in range(len(sub) - 1):
                print(
                    f"  segment D={D:g} Rc={R_tail:g} {parity} "
                    f"R={sub.iloc[idx]['R']:g}->{sub.iloc[idx + 1]['R']:g}",
                    flush=True,
                )
                segment_rows.append(
                    pair_audit(
                        row_a=sub.iloc[idx],
                        row_b=sub.iloc[idx + 1],
                        pad_factor=float(pad_factor),
                        n_per_side=int(args.n_per_side),
                        cfg=cfg,
                        p=p,
                        contour_kind="segment",
                    )
                )
            if not args.skip_full_path:
                print(f"  full path D={D:g} Rc={R_tail:g} {parity}", flush=True)
                path_rows.extend(
                    full_path_audit(
                        sub=sub,
                        pad_factor=float(pad_factor),
                        n_per_side=int(args.n_per_side),
                        cfg=cfg,
                        p=p,
                    )
                )

    segment_df = pd.DataFrame(segment_rows)
    path_df = pd.DataFrame(path_rows)
    summary = summarize(segment=segment_df, path=path_df)
    adaptive_summary = adaptive_summarize(segment=segment_df, path=path_df)

    segment_out = OUTDIR / f"{args.out_tag}_segment_detail.csv"
    path_out = OUTDIR / f"{args.out_tag}_path_detail.csv"
    summary_out = OUTDIR / f"{args.out_tag}_summary.csv"
    adaptive_out = OUTDIR / f"{args.out_tag}_adaptive_summary.csv"
    segment_df.to_csv(segment_out, index=False)
    path_df.to_csv(path_out, index=False)
    summary.to_csv(summary_out, index=False)
    adaptive_summary.to_csv(adaptive_out, index=False)

    plot_out = OUTDIR / f"{args.out_tag}_budget.png"
    if not args.skip_plot:
        plot_budget(summary=summary, out_path=plot_out)

    print(f"Wrote {segment_out}")
    print(f"Wrote {path_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {adaptive_out}")
    if not args.skip_plot:
        print(f"Wrote {plot_out}")
    cols = [
        "tube_kind",
        "pad_factor",
        "n_segments",
        "n_endpoint_index_ok",
        "n_rouche_pass_from_R0",
        "max_rouche_ratio_from_R0",
        "median_rouche_ratio_from_R0",
        "min_boundary_F_abs",
        "large_tube_status",
    ]
    print("[Large-tube Rouché summary]")
    print(summary[cols].to_string(index=False))
    if len(adaptive_summary) > 0:
        print("[Adaptive large-tube Rouché summary]")
        print(adaptive_summary.to_string(index=False))


if __name__ == "__main__":
    main()

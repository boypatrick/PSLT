#!/usr/bin/env python3
"""
Finite-difference R-flow audit for the static-width S16 gate.

This is the first theorem-facing diagnostic after the S15 common-box failure.
It does not solve for new roots, does not introduce a new backend, and does not
change the Whittaker-DtN family.  It asks how large the parent-side
zero-preserving R-flow must be if the already-certified S8 local zeros are to
be connected into one moving pole branch.

For each fixed (D, R_tail, parity) branch label, the script sorts the existing
Whittaker roots by interface radius R and records the finite-difference slope

    dE/dR ~= (E(R_{j+1}) - E(R_j)) / (R_{j+1} - R_j).

The resulting path length and segment displacements are normalized by the S8
local-box half-widths.  Values much larger than one do not disprove an analytic
R-homotopy theorem, but they show that the theorem cannot be a tiny local-box
perturbation: it must transport the zero through a larger pole-free tube.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_r_flow_bound_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_r_flow_bound_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_r_flow_bound_budget.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_CONTOUR_JOST = OUTDIR / "static_width_whittaker_contour_jost_bound_detail.csv"
DEFAULT_TAG = "static_width_whittaker_r_flow_bound"


def contour_lookup(contour: pd.DataFrame) -> dict[tuple[float, float, float, str], pd.Series]:
    out: dict[tuple[float, float, float, str], pd.Series] = {}
    for _, row in contour.iterrows():
        key = (float(row["D"]), float(row["R"]), float(row["R_tail_root"]), str(row["parity"]))
        out[key] = row
    return out


def endpoint_tail_stats(
    p0: pd.Series,
    p1: pd.Series,
    contour_by_key: dict[tuple[float, float, float, str], pd.Series],
) -> dict[str, float]:
    rows: list[pd.Series] = []
    for point in (p0, p1):
        key = (float(point["D"]), float(point["R"]), float(point["R_tail"]), str(point["parity"]))
        if key in contour_by_key:
            rows.append(contour_by_key[key])
    if not rows:
        return {
            "endpoint_tail_box_pass_count": 0.0,
            "endpoint_boundary_ratio_max": np.nan,
            "endpoint_product_margin_min": np.nan,
            "endpoint_winding_abs_max": np.nan,
        }
    data = pd.DataFrame(rows)
    pass_count = float((data["contour_jost_status"] == "passes_contour_jost_box").sum())
    winding_abs = max(
        float(data["jost_a_winding_round"].abs().max()),
        float(data["jost_b_winding_round"].abs().max()),
    )
    return {
        "endpoint_tail_box_pass_count": pass_count,
        "endpoint_boundary_ratio_max": float(data["boundary_ratio_direct_max"].max()),
        "endpoint_product_margin_min": float(data["boundary_product_margin_min"].min()),
        "endpoint_winding_abs_max": winding_abs,
    }


def group_square_cover(sub: pd.DataFrame) -> float:
    re = sub["E0_real"].to_numpy(dtype=float)
    im = sub["E0_imag"].to_numpy(dtype=float)
    box_re = float(sub["box_re"].min())
    box_im = float(sub["box_im"].min())
    return float(
        max(
            (np.max(re) - np.min(re)) / max(2.0 * box_re, 1.0e-300),
            (np.max(im) - np.min(im)) / max(2.0 * box_im, 1.0e-300),
        )
    )


def segment_rows(argument: pd.DataFrame, contour: pd.DataFrame) -> pd.DataFrame:
    contour_by_key = contour_lookup(contour)
    rows: list[dict[str, float | str]] = []
    group_cols = ["D", "R_tail", "parity"]
    for (D, R_tail, parity), sub in argument.groupby(group_cols, sort=True):
        sub = sub.sort_values("R").reset_index(drop=True)
        if len(sub) < 2:
            continue
        cover = group_square_cover(sub)
        for idx in range(len(sub) - 1):
            a = sub.iloc[idx]
            b = sub.iloc[idx + 1]
            E0 = complex(float(a["E0_real"]), float(a["E0_imag"]))
            E1 = complex(float(b["E0_real"]), float(b["E0_imag"]))
            dE = E1 - E0
            dR = float(b["R"]) - float(a["R"])
            box_re = min(float(a["box_re"]), float(b["box_re"]))
            box_im = min(float(a["box_im"]), float(b["box_im"]))
            box_disk = min(box_re, box_im)
            step_square = max(
                abs(dE.real) / max(2.0 * box_re, 1.0e-300),
                abs(dE.imag) / max(2.0 * box_im, 1.0e-300),
            )
            step_disk = abs(dE) / max(2.0 * box_disk, 1.0e-300)
            stats = endpoint_tail_stats(a, b, contour_by_key)
            rows.append(
                {
                    "D": float(D),
                    "R_tail": float(R_tail),
                    "parity": str(parity),
                    "R0": float(a["R"]),
                    "R1": float(b["R"]),
                    "dR": dR,
                    "E0_real": float(E0.real),
                    "E0_imag": float(E0.imag),
                    "E1_real": float(E1.real),
                    "E1_imag": float(E1.imag),
                    "dE_real": float(dE.real),
                    "dE_imag": float(dE.imag),
                    "dE_abs": float(abs(dE)),
                    "finite_flow_abs_per_R": float(abs(dE) / max(abs(dR), 1.0e-300)),
                    "finite_flow_re_per_R": float(dE.real / dR),
                    "finite_flow_im_per_R": float(dE.imag / dR),
                    "box_re_min": box_re,
                    "box_im_min": box_im,
                    "step_square_cover_factor": float(step_square),
                    "step_disk_cover_factor": float(step_disk),
                    "group_square_cover_factor": cover,
                    **stats,
                    "segment_status": "tail_box_control_but_large_R_flow"
                    if stats["endpoint_tail_box_pass_count"] == 2.0 and step_square > 1.0
                    else (
                        "tail_box_control_and_step_boxes_overlap"
                        if stats["endpoint_tail_box_pass_count"] == 2.0
                        else "missing_tail_box_control"
                    ),
                }
            )
    return pd.DataFrame(rows)


def path_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    group_cols = ["D", "R_tail", "parity"]
    for (D, R_tail, parity), sub in detail.groupby(group_cols, sort=True):
        sub = sub.sort_values("R0")
        path_length = float(sub["dE_abs"].sum())
        net = complex(
            float(sub["E1_real"].iloc[-1] - sub["E0_real"].iloc[0]),
            float(sub["E1_imag"].iloc[-1] - sub["E0_imag"].iloc[0]),
        )
        box_half = float(min(sub["box_re_min"].min(), sub["box_im_min"].min()))
        tail_ok = (
            (sub["endpoint_tail_box_pass_count"] == 2.0)
            & (sub["endpoint_boundary_ratio_max"] < 1.0)
            & (sub["endpoint_winding_abs_max"] == 0.0)
        )
        rows.append(
            {
                "D": float(D),
                "R_tail": float(R_tail),
                "parity": str(parity),
                "n_segments": float(len(sub)),
                "R_min": float(sub["R0"].min()),
                "R_max": float(sub["R1"].max()),
                "path_length": path_length,
                "net_displacement_abs": float(abs(net)),
                "path_over_net": float(path_length / max(abs(net), 1.0e-300)),
                "path_length_over_box_halfwidth": float(path_length / max(box_half, 1.0e-300)),
                "max_step_abs": float(sub["dE_abs"].max()),
                "max_step_square_cover_factor": float(sub["step_square_cover_factor"].max()),
                "max_step_disk_cover_factor": float(sub["step_disk_cover_factor"].max()),
                "group_square_cover_factor": float(sub["group_square_cover_factor"].max()),
                "mean_flow_abs_per_R": float(path_length / max(float(sub["R1"].max() - sub["R0"].min()), 1.0e-300)),
                "max_flow_abs_per_R": float(sub["finite_flow_abs_per_R"].max()),
                "endpoint_boundary_ratio_max": float(sub["endpoint_boundary_ratio_max"].max()),
                "endpoint_product_margin_min": float(sub["endpoint_product_margin_min"].min()),
                "all_endpoint_tail_box_control": float(tail_ok.all()),
                "path_status": "tail_box_control_closed_but_R_flow_needs_large_tube"
                if bool(tail_ok.all()) and float(sub["step_square_cover_factor"].max()) > 1.0
                else (
                    "local_step_boxes_overlap"
                    if bool(tail_ok.all())
                    else "missing_tail_box_control"
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in paths.groupby("D", sort=True):
        rows.append(summary_row(D=float(D), sub=sub))
    rows.append(summary_row(D="global", sub=paths))
    return pd.DataFrame(rows)


def summary_row(D: float | str, sub: pd.DataFrame) -> dict[str, float | str]:
    n_tail = float((sub["all_endpoint_tail_box_control"] > 0.5).sum())
    n_step_overlap = float((sub["max_step_square_cover_factor"] <= 1.0).sum())
    status = "R_flow_large_tube_required"
    if n_tail == len(sub) and n_step_overlap == len(sub):
        status = "local_R_flow_boxes_overlap"
    elif n_tail < len(sub):
        status = "missing_tail_box_control"
    return {
        "D": D,
        "n_paths": float(len(sub)),
        "n_paths_with_tail_box_control": n_tail,
        "n_paths_with_step_box_overlap": n_step_overlap,
        "min_path_length_over_box_halfwidth": float(sub["path_length_over_box_halfwidth"].min()),
        "median_path_length_over_box_halfwidth": float(sub["path_length_over_box_halfwidth"].median()),
        "max_path_length_over_box_halfwidth": float(sub["path_length_over_box_halfwidth"].max()),
        "min_group_square_cover_factor": float(sub["group_square_cover_factor"].min()),
        "median_group_square_cover_factor": float(sub["group_square_cover_factor"].median()),
        "max_group_square_cover_factor": float(sub["group_square_cover_factor"].max()),
        "min_endpoint_product_margin": float(sub["endpoint_product_margin_min"].min()),
        "max_endpoint_boundary_ratio": float(sub["endpoint_boundary_ratio_max"].max()),
        "max_mean_flow_abs_per_R": float(sub["mean_flow_abs_per_R"].max()),
        "max_step_square_cover_factor": float(sub["max_step_square_cover_factor"].max()),
        "r_flow_status": status,
    }


def plot_budget(paths: pd.DataFrame, out_path: Path) -> None:
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/pslt_matplotlib_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/pslt_xdg_cache"))
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"D={int(row.D)} {row.parity} Rc={int(row.R_tail)}" for row in paths.itertuples()]
    x = np.arange(len(paths))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - 0.18, paths["group_square_cover_factor"], width=0.36, label="net square cover")
    ax.bar(x + 0.18, paths["path_length_over_box_halfwidth"], width=0.36, label="path length / box halfwidth")
    ax.axhline(1.0, color="black", lw=1.0, ls="--", label="local-box threshold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("normalized R-flow budget")
    ax.set_title("S16 finite-difference R-flow tube requirement")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT)
    ap.add_argument("--contour-jost-detail", type=Path, default=DEFAULT_CONTOUR_JOST)
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    argument = pd.read_csv(args.argument_detail)
    contour = pd.read_csv(args.contour_jost_detail)
    detail = segment_rows(argument=argument, contour=contour)
    paths = path_summary(detail)
    summary = summarize(paths)

    detail_path = OUTDIR / f"{args.out_tag}_detail.csv"
    paths_path = OUTDIR / f"{args.out_tag}_paths.csv"
    summary_path = OUTDIR / f"{args.out_tag}_summary.csv"
    detail.to_csv(detail_path, index=False)
    paths.to_csv(paths_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {paths_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))

    if not args.skip_plot:
        plot_path = OUTDIR / f"{args.out_tag}_budget.png"
        plot_budget(paths, plot_path)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_auxiliary_extraction_audit import (
    N_INNER,
    N_OUTER_EACH,
    N_TOTAL,
    PAPER_DIR,
    P,
    OUTDIR,
    Z_MAX,
    build_prolate_axis_mesh,
    build_uniform_axis_mesh,
    solve_nonuniform_bound_states,
)


STEM = "prolate_calibration_vs_2d_reference_audit"

D_GRID = np.array([6.0, 12.0, 18.0], dtype=float)

OMEGA_EXACT_PATH = OUTDIR.parent / "omega_fp_1d" / "omega_exact_D6-12-18.csv"
CHI_LOCALIZED_PATH = OUTDIR.parent / "chi_fp_2d" / "localized_chi_D6-12-18.csv"
CHI_FP2D_PATH = OUTDIR.parent / "chi_fp_2d" / "chi_fp2d_D6-12-18.csv"


def _solve_auxiliary(D: float) -> dict[str, float]:
    z_uniform = build_uniform_axis_mesh(Z_MAX, N_TOTAL)
    z_prolate = build_prolate_axis_mesh(float(D), Z_MAX, N_INNER, N_OUTER_EACH)

    evals_uniform, omega_uniform = solve_nonuniform_bound_states(z_uniform, float(D), P, n_states=3)
    evals_prolate, omega_prolate = solve_nonuniform_bound_states(z_prolate, float(D), P, n_states=3)

    return {
        "D": float(D),
        "uniform_E1": float(evals_uniform[0]),
        "uniform_E2": float(evals_uniform[1]),
        "uniform_omega1": float(omega_uniform[0]),
        "uniform_omega2": float(omega_uniform[1]),
        "uniform_delta_omega12": float(omega_uniform[1] - omega_uniform[0]),
        "prolate_E1": float(evals_prolate[0]),
        "prolate_E2": float(evals_prolate[1]),
        "prolate_omega1": float(omega_prolate[0]),
        "prolate_omega2": float(omega_prolate[1]),
        "prolate_delta_omega12": float(omega_prolate[1] - omega_prolate[0]),
        "uniform_n_total": int(len(z_uniform)),
        "prolate_n_total": int(len(z_prolate)),
        "prolate_n_inner": int(N_INNER),
        "prolate_n_outer_each": int(N_OUTER_EACH),
    }


def _rel_err(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    cur = np.asarray(cur, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return np.abs(cur - ref) / np.maximum(np.abs(ref), 1e-30)


def _anchor_ratio(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    anchor = values[0]
    return values / anchor if abs(anchor) > 1e-30 else values.copy()


def _affine_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    scale = float(np.std(centered))
    return centered / scale if scale > 1e-30 else centered.copy()


def _shape_distance(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    diff = lhs - rhs
    return {
        "l1": float(np.sum(np.abs(diff))),
        "l2": float(np.sqrt(np.sum(diff**2))),
        "linf": float(np.max(np.abs(diff))),
    }


def _build_detail() -> pd.DataFrame:
    exact = pd.read_csv(OMEGA_EXACT_PATH)
    exact = exact.loc[exact["level"] == "fine"].copy().set_index("D")

    chi_loc = pd.read_csv(CHI_LOCALIZED_PATH)
    chi_loc = chi_loc.loc[chi_loc["level"] == "fine"].copy().set_index("D")

    chi_fp2d = pd.read_csv(CHI_FP2D_PATH)
    chi_fp2d = chi_fp2d.loc[chi_fp2d["level"] == "fine"].copy().set_index("D")

    rows = []
    for D in D_GRID:
        aux = _solve_auxiliary(float(D))
        ex = exact.loc[D]
        loc = chi_loc.loc[D]
        fp2d = chi_fp2d.loc[D]

        row = {
            **aux,
            "exact_fine_E1": float(ex["E1"]),
            "exact_fine_E2": float(ex["E2"]),
            "exact_fine_omega1": float(ex["omega1"]),
            "exact_fine_omega2": float(ex["omega2"]),
            "exact_fine_delta_omega12": float(ex["delta_omega12"]),
            "localized_fine_E1": float(loc["E1"]),
            "localized_fine_E2": float(loc["E2"]),
            "localized_fine_omega1": float(loc["omega1"]),
            "localized_fine_delta_proxy": float(loc["DeltaE"]),
            "chi_fp2d_fine_E1": float(fp2d["E1"]),
            "chi_fp2d_fine_E2": float(fp2d["E2"]),
            "chi_fp2d_fine_omega1": float(fp2d["omega1"]),
            "chi_fp2d_fine_omega2": float(fp2d["omega2"]),
            "chi_fp2d_fine_delta_proxy": float(fp2d["omega2"] - fp2d["omega1"]),
        }
        for metric in ["E1", "E2", "omega1", "omega2", "delta_omega12"]:
            row[f"uniform_rel_{metric}_vs_exact_fine"] = float(
                _rel_err(np.array([row[f"uniform_{metric}"]]), np.array([row[f"exact_fine_{metric}"]]))[0]
            )
            row[f"prolate_rel_{metric}_vs_exact_fine"] = float(
                _rel_err(np.array([row[f"prolate_{metric}"]]), np.array([row[f"exact_fine_{metric}"]]))[0]
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _build_exact_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["E1", "E2", "omega1", "omega2", "delta_omega12"]:
        u = detail[f"uniform_rel_{metric}_vs_exact_fine"].to_numpy(dtype=float)
        p = detail[f"prolate_rel_{metric}_vs_exact_fine"].to_numpy(dtype=float)
        rows.append(
            {
                "summary_type": "exact_target",
                "target": "omega_exact_fine",
                "metric": metric,
                "uniform_p95_rel_err": float(np.percentile(u, 95.0)),
                "prolate_p95_rel_err": float(np.percentile(p, 95.0)),
                "uniform_max_rel_err": float(np.max(u)),
                "prolate_max_rel_err": float(np.max(p)),
                "prolate_over_uniform_p95_ratio": float(np.percentile(p, 95.0) / max(np.percentile(u, 95.0), 1e-30)),
                "better_method_by_p95": "prolate_aux" if np.percentile(p, 95.0) < np.percentile(u, 95.0) else "uniform_aux",
            }
        )
    return pd.DataFrame(rows)


def _build_witness_shape_summary(detail: pd.DataFrame) -> pd.DataFrame:
    witness_specs = [
        ("localized_fine", "omega1", "uniform_omega1", "prolate_omega1"),
        ("localized_fine", "delta_proxy", "uniform_delta_omega12", "prolate_delta_omega12"),
        ("chi_fp2d_fine", "omega1", "uniform_omega1", "prolate_omega1"),
        ("chi_fp2d_fine", "omega2", "uniform_omega2", "prolate_omega2"),
        ("chi_fp2d_fine", "delta_proxy", "uniform_delta_omega12", "prolate_delta_omega12"),
    ]

    rows = []
    for witness_prefix, witness_metric, uniform_col, prolate_col in witness_specs:
        witness = detail[f"{witness_prefix}_{witness_metric}"].to_numpy(dtype=float)
        uniform = detail[uniform_col].to_numpy(dtype=float)
        prolate = detail[prolate_col].to_numpy(dtype=float)

        witness_anchor = _anchor_ratio(witness)
        uniform_anchor = _anchor_ratio(uniform)
        prolate_anchor = _anchor_ratio(prolate)

        witness_affine = _affine_profile(witness)
        uniform_affine = _affine_profile(uniform)
        prolate_affine = _affine_profile(prolate)

        u_anchor = _shape_distance(uniform_anchor, witness_anchor)
        p_anchor = _shape_distance(prolate_anchor, witness_anchor)
        u_affine = _shape_distance(uniform_affine, witness_affine)
        p_affine = _shape_distance(prolate_affine, witness_affine)

        rows.append(
            {
                "summary_type": "witness_shape",
                "witness": f"{witness_prefix}_{witness_metric}",
                "uniform_anchor_l1": u_anchor["l1"],
                "prolate_anchor_l1": p_anchor["l1"],
                "uniform_anchor_linf": u_anchor["linf"],
                "prolate_anchor_linf": p_anchor["linf"],
                "uniform_affine_l1": u_affine["l1"],
                "prolate_affine_l1": p_affine["l1"],
                "uniform_affine_linf": u_affine["linf"],
                "prolate_affine_linf": p_affine["linf"],
                "better_method_anchor": "prolate_aux" if p_anchor["l1"] < u_anchor["l1"] else "uniform_aux",
                "better_method_affine": "prolate_aux" if p_affine["l1"] < u_affine["l1"] else "uniform_aux",
            }
        )
    return pd.DataFrame(rows)


def _build_slices(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in detail.iterrows():
        rows.append(
            {
                "D": float(row["D"]),
                "exact_fine_omega1": float(row["exact_fine_omega1"]),
                "uniform_omega1": float(row["uniform_omega1"]),
                "prolate_omega1": float(row["prolate_omega1"]),
                "exact_fine_omega2": float(row["exact_fine_omega2"]),
                "uniform_omega2": float(row["uniform_omega2"]),
                "prolate_omega2": float(row["prolate_omega2"]),
                "exact_fine_delta_omega12": float(row["exact_fine_delta_omega12"]),
                "uniform_delta_omega12": float(row["uniform_delta_omega12"]),
                "prolate_delta_omega12": float(row["prolate_delta_omega12"]),
                "localized_fine_omega1": float(row["localized_fine_omega1"]),
                "localized_fine_delta_proxy": float(row["localized_fine_delta_proxy"]),
                "chi_fp2d_fine_omega1": float(row["chi_fp2d_fine_omega1"]),
                "chi_fp2d_fine_omega2": float(row["chi_fp2d_fine_omega2"]),
                "chi_fp2d_fine_delta_proxy": float(row["chi_fp2d_fine_delta_proxy"]),
                "uniform_rel_omega1_vs_exact_fine": float(row["uniform_rel_omega1_vs_exact_fine"]),
                "prolate_rel_omega1_vs_exact_fine": float(row["prolate_rel_omega1_vs_exact_fine"]),
                "uniform_rel_omega2_vs_exact_fine": float(row["uniform_rel_omega2_vs_exact_fine"]),
                "prolate_rel_omega2_vs_exact_fine": float(row["prolate_rel_omega2_vs_exact_fine"]),
                "uniform_rel_delta_omega12_vs_exact_fine": float(row["uniform_rel_delta_omega12_vs_exact_fine"]),
                "prolate_rel_delta_omega12_vs_exact_fine": float(row["prolate_rel_delta_omega12_vs_exact_fine"]),
            }
        )
    return pd.DataFrame(rows)


def _plot(detail: pd.DataFrame, out_png: Path) -> None:
    x = detail["D"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, detail["exact_fine_omega1"], marker="o", lw=2, label="1D exact fine")
    ax.plot(x, detail["uniform_omega1"], marker="s", lw=2, label="uniform aux")
    ax.plot(x, detail["prolate_omega1"], marker="^", lw=2, label="prolate aux")
    ax.set_title("omega1 calibration")
    ax.set_xlabel("D")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, detail["exact_fine_delta_omega12"], marker="o", lw=2, label="1D exact fine")
    ax.plot(x, detail["uniform_delta_omega12"], marker="s", lw=2, label="uniform aux")
    ax.plot(x, detail["prolate_delta_omega12"], marker="^", lw=2, label="prolate aux")
    ax.set_title("delta omega12 calibration")
    ax.set_xlabel("D")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, _anchor_ratio(detail["uniform_omega1"].to_numpy(dtype=float)), marker="s", lw=2, label="uniform aux")
    ax.plot(x, _anchor_ratio(detail["prolate_omega1"].to_numpy(dtype=float)), marker="^", lw=2, label="prolate aux")
    ax.plot(
        x,
        _anchor_ratio(detail["localized_fine_omega1"].to_numpy(dtype=float)),
        marker="o",
        lw=2,
        label="localized 2D witness",
    )
    ax.plot(
        x,
        _anchor_ratio(detail["chi_fp2d_fine_omega1"].to_numpy(dtype=float)),
        marker="d",
        lw=2,
        label="chi_fp2d witness",
    )
    ax.set_title("Anchor-normalized omega1 trend")
    ax.set_xlabel("D")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(
        x,
        _anchor_ratio(detail["uniform_delta_omega12"].to_numpy(dtype=float)),
        marker="s",
        lw=2,
        label="uniform aux",
    )
    ax.plot(
        x,
        _anchor_ratio(detail["prolate_delta_omega12"].to_numpy(dtype=float)),
        marker="^",
        lw=2,
        label="prolate aux",
    )
    ax.plot(
        x,
        _anchor_ratio(detail["localized_fine_delta_proxy"].to_numpy(dtype=float)),
        marker="o",
        lw=2,
        label="localized 2D witness",
    )
    ax.plot(
        x,
        _anchor_ratio(detail["chi_fp2d_fine_delta_proxy"].to_numpy(dtype=float)),
        marker="d",
        lw=2,
        label="chi_fp2d witness",
    )
    ax.set_title("Anchor-normalized splitting trend")
    ax.set_xlabel("D")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    detail = _build_detail()
    exact_summary = _build_exact_summary(detail)
    witness_summary = _build_witness_shape_summary(detail)
    summary = pd.concat([exact_summary, witness_summary], ignore_index=True, sort=False)
    slices = _build_slices(detail)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(detail, png_path)

    meta_path.write_text(
        json.dumps(
            {
                "D_grid": [float(x) for x in D_GRID],
                "auxiliary_mesh": {
                    "uniform_n_total": N_TOTAL,
                    "prolate_n_total": N_TOTAL,
                    "prolate_n_inner": N_INNER,
                    "prolate_n_outer_each": N_OUTER_EACH,
                    "z_max": Z_MAX,
                },
                "physical_params": {
                    "a": P.a,
                    "eps": P.eps,
                    "m0": P.m0,
                    "xi": P.xi,
                },
                "targets": {
                    "absolute_target": str(OMEGA_EXACT_PATH),
                    "witnesses": [str(CHI_LOCALIZED_PATH), str(CHI_FP2D_PATH)],
                },
                "notes": [
                    "omega_exact_D6-12-18 fine is the only absolute calibration target.",
                    "localized_chi and chi_fp2d are witness-only trend references, not the same eigenproblem as bound-state omega.",
                ],
            },
            indent=2,
        )
    )

    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
